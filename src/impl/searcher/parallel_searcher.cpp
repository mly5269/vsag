
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "parallel_searcher.h"

#include <limits>
#include <utility>

#include "impl/heap/standard_heap.h"
#include "utils/linear_congruential_generator.h"

namespace vsag {

ParallelSearcher::ParallelSearcher(const IndexCommonParam& common_param,
                                   std::shared_ptr<SafeThreadPool> search_pool,
                                   MutexArrayPtr mutex_array)
    : allocator_(common_param.allocator_.get()),
      pool(std::move(search_pool)),
      mutex_array_(std::move(mutex_array)) {
    vt_mutex_array = std::make_shared<PointsMutex>(NUM_STRIPES,allocator_);
}

uint32_t
ParallelSearcher::visit(const GraphInterfacePtr& graph,
                        const VisitedListPtr& vl,
                        const Vector<std::pair<float, uint64_t>>& node_pair,
                        const FilterPtr& filter,
                        float skip_ratio,
                        Vector<InnerIdType>& to_be_visited_rid,
                        Vector<InnerIdType>& to_be_visited_id,
                        std::vector<Vector<InnerIdType>>& neighbors,
                        uint64_t point_visited_num) const {
    LinearCongruentialGenerator generator;
    uint32_t count_no_visited = 0;

    if (this->mutex_array_ != nullptr) {
        for (uint64_t i = 0; i < point_visited_num; i++) {
            SharedLock lock(this->mutex_array_, node_pair[i].second);
            graph->GetNeighbors(node_pair[i].second, neighbors[i]);
        }
    } else {
        for (uint64_t i = 0; i < point_visited_num; i++) {
            graph->GetNeighbors(node_pair[i].second, neighbors[i]);
        }
    }

    float skip_threshold =
        (filter != nullptr
             ? (filter->ValidRatio() == 1.0F ? 0 : (1 - ((1 - filter->ValidRatio()) * skip_ratio)))
             : 0.0F);
    for (uint64_t i = 0; i < point_visited_num; i++) {
        for (uint32_t j = 0; j < neighbors[i].size(); j++) {
            if (j + prefetch_stride_visit_ < neighbors[i].size()) {
                vl->Prefetch(neighbors[i][j + prefetch_stride_visit_]);
            }
            if (not vl->Get(neighbors[i][j])) {
                if (not filter || count_no_visited == 0 || generator.NextFloat() > skip_threshold ||
                    filter->CheckValid(neighbors[i][j])) {
                    to_be_visited_rid[count_no_visited] = j;
                    to_be_visited_id[count_no_visited] = neighbors[i][j];
                    count_no_visited++;
                }
                vl->Set(neighbors[i][j]);
            }
        }
    }
    return count_no_visited;
}

uint32_t
ParallelSearcher::parallel_visit(const GraphInterfacePtr& graph,
                     const VisitedListPtr& vl,
                     const std::pair<float, uint64_t>& current_node_pair,
                     const FilterPtr& filter,
                     float skip_ratio,
                     Vector<InnerIdType>& to_be_visited_rid,
                     Vector<InnerIdType>& to_be_visited_id,
                     Vector<InnerIdType>& neighbors) const {
    LinearCongruentialGenerator generator;
    uint32_t count_no_visited = 0;

    if (this->mutex_array_ != nullptr) {
        SharedLock lock(this->mutex_array_, current_node_pair.second);
        graph->GetNeighbors(current_node_pair.second, neighbors);
    } else {
        graph->GetNeighbors(current_node_pair.second, neighbors);
    }

    float skip_threshold =
        (filter != nullptr
             ? (filter->ValidRatio() == 1.0F ? 0 : (1 - ((1 - filter->ValidRatio()) * skip_ratio)))
             : 0.0F);

    for (uint32_t i = 0; i < neighbors.size(); i++) {
        uint32_t lock_piece = neighbors[i] & (NUM_STRIPES - 1);
        vt_mutex_array->SharedLock(lock_piece);
        if (i + prefetch_stride_visit_ < neighbors.size()) {//这里就先这样，之后再考虑是否减小锁的范围，先以快速实现为主
            vl->Prefetch(neighbors[i + prefetch_stride_visit_]);
        }
        bool skip = vl->Get(neighbors[i]);
        vt_mutex_array->SharedUnlock(lock_piece);
        if (not skip) {
            vt_mutex_array->Lock(lock_piece);
            vl->Set(neighbors[i]);
            vt_mutex_array->Unlock(lock_piece);
            if (not filter || count_no_visited == 0 || generator.NextFloat() > skip_threshold ||
                filter->CheckValid(neighbors[i])) {
                to_be_visited_rid[count_no_visited] = i;
                to_be_visited_id[count_no_visited] = neighbors[i];
                count_no_visited++;
            }
        }
    }
    return count_no_visited;

    // LinearCongruentialGenerator generator;
    // uint32_t count_no_visited = 0;

    // if (this->mutex_array_ != nullptr) {
    //     SharedLock lock(this->mutex_array_, current_node_pair.second);
    //     graph->GetNeighbors(current_node_pair.second, neighbors);
    // } else {
    //     graph->GetNeighbors(current_node_pair.second, neighbors);
    // }

    // float skip_threshold =
    //     (filter != nullptr
    //          ? (filter->ValidRatio() == 1.0F ? 0 : (1 - ((1 - filter->ValidRatio()) * skip_ratio)))
    //          : 0.0F);

    // for (uint32_t i = 0; i < neighbors.size(); i++) {
    //     if (i + prefetch_stride_visit_ < neighbors.size()) {
    //         vl->Prefetch(neighbors[i + prefetch_stride_visit_]);
    //     }
    //     if (not vl->Get(neighbors[i])) {
    //         if (not filter || count_no_visited == 0 || generator.NextFloat() > skip_threshold ||
    //             filter->CheckValid(neighbors[i])) {
    //             to_be_visited_rid[count_no_visited] = i;
    //             to_be_visited_id[count_no_visited] = neighbors[i];
    //             count_no_visited++;
    //         }
    //         vl->Set(neighbors[i]);
    //     }
    // }
    // return count_no_visited;
}

DistHeapPtr
ParallelSearcher::Search(const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         const VisitedListPtr& vl,
                         const void* query,
                         const InnerSearchParam& inner_search_param,
                         const LabelTablePtr& label_table) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(
            graph, flatten, vl, query, inner_search_param, label_table);
    }
    return this->search_impl<RANGE_SEARCH>(
        graph, flatten, vl, query, inner_search_param, label_table);
}

DistHeapPtr
ParallelSearcher::Parallel_Search(const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         const VisitedListPtr& vl,
                         const void* query,
                         const InnerSearchParam& inner_search_param,
                         const LabelTablePtr& label_table) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->parallel_search_impl<KNN_SEARCH>(
            graph, flatten, vl, query, inner_search_param, label_table);
    }
    return this->parallel_search_impl<RANGE_SEARCH>(
        graph, flatten, vl, query, inner_search_param, label_table);
}

template <InnerSearchMode mode>
DistHeapPtr
ParallelSearcher::search_impl(const GraphInterfacePtr& graph,
                              const FlattenInterfacePtr& flatten,
                              const VisitedListPtr& vl,
                              const void* query,
                              const InnerSearchParam& inner_search_param,
                              const LabelTablePtr& label_table) const {
    Allocator* alloc =
        inner_search_param.search_alloc == nullptr ? allocator_ : inner_search_param.search_alloc;
    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto computer = flatten->FactoryComputer(query);

    auto is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;

    float dist = 0.0F;
    auto lower_bound = std::numeric_limits<float>::max();

    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    uint32_t vector_size =
        graph->MaximumDegree() * inner_search_param.parallel_search_thread_count_per_query;
    uint32_t current_start = 0;
    Vector<InnerIdType> to_be_visited_rid(vector_size, alloc);
    Vector<InnerIdType> to_be_visited_id(vector_size, alloc);
    std::vector<Vector<InnerIdType>> neighbors(
        inner_search_param.parallel_search_thread_count_per_query,
        Vector<InnerIdType>(graph->MaximumDegree(), alloc));
    Vector<float> line_dists(vector_size, alloc);
    Vector<std::pair<float, uint64_t>> node_pair(
        inner_search_param.parallel_search_thread_count_per_query, alloc);
    Vector<uint32_t> tasks_per_thread(inner_search_param.parallel_search_thread_count_per_query,
                                      alloc);
    Vector<uint32_t> start_index(inner_search_param.parallel_search_thread_count_per_query, alloc);

    flatten->Query(&dist, computer, &ep, 1, alloc);
    if (not is_id_allowed || is_id_allowed->CheckValid(ep)) {
        top_candidates->Push(dist, ep);
        lower_bound = top_candidates->Top().first;
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (dist > inner_search_param.radius and not top_candidates->Empty()) {
            top_candidates->Pop();
        }
    }
    if (dist < THRESHOLD_ERROR) {
        inner_search_param.duplicate_id = ep;
    }
    candidate_set->Push(-dist, ep);
    vl->Set(ep);

    while (not candidate_set->Empty()) {
        hops++;
        auto num_explore_nodes =
            candidate_set->Size() < inner_search_param.parallel_search_thread_count_per_query
                ? candidate_set->Size()
                : inner_search_param.parallel_search_thread_count_per_query;

        auto current_first_node_pair = candidate_set->Top();
        node_pair[0] = current_first_node_pair;

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_first_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                break;
            }
        }
        candidate_set->Pop();

        for (uint64_t i = 1; i < num_explore_nodes; i++) {
            node_pair[i] = candidate_set->Top();
            candidate_set->Pop();
        }

        count_no_visited = visit(graph,
                                 vl,
                                 node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 to_be_visited_rid,
                                 to_be_visited_id,
                                 neighbors,
                                 num_explore_nodes);

        dist_cmp += count_no_visited;
        uint64_t num_threads = num_explore_nodes;

        uint32_t base = 0;
        uint32_t remainder = 0;

        if (num_threads) {
            base = count_no_visited / num_threads;
            remainder = count_no_visited % num_threads;
        }

        current_start = 0;
        for (uint64_t i = 0; i < num_threads; ++i) {
            tasks_per_thread[i] = base + (i < remainder ? 1 : 0);
            start_index[i] = current_start;
            current_start += tasks_per_thread[i];
        }

        auto dist_compute = [&](uint64_t i) -> void {
            flatten->Query(line_dists.data() + start_index[i],
                           computer,
                           to_be_visited_id.data() + start_index[i],
                           tasks_per_thread[i],
                           alloc);
        };

        std::vector<std::future<void>> futures;

        for (uint64_t i = 0; i < num_threads; i++) {
            futures.emplace_back(pool->GeneralEnqueue(dist_compute, i));
        }

        for (auto& f : futures) {
            f.get();
        }

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (dist < THRESHOLD_ERROR) {
                inner_search_param.duplicate_id = to_be_visited_id[i];
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                candidate_set->Push(-dist, to_be_visited_id[i]);
                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited_id[i])) {
                    top_candidates->Push(dist, to_be_visited_id[i]);
                }
                if (inner_search_param.consider_duplicate && label_table &&
                    label_table->CompressDuplicateData()) {
                    const auto& duplicate_ids = label_table->GetDuplicateId(to_be_visited_id[i]);
                    for (const auto& item : duplicate_ids) {
                        top_candidates->Push(dist, item);
                    }
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates->Size() > ef) {
                        top_candidates->Pop();
                    }
                }

                if (not top_candidates->Empty()) {
                    lower_bound = top_candidates->Top().first;
                }
            }
        }
    }

    if constexpr (mode == KNN_SEARCH) {
        while (top_candidates->Size() > inner_search_param.topk) {
            top_candidates->Pop();
        }
    } else if constexpr (mode == RANGE_SEARCH) {
        if (inner_search_param.range_search_limit_size > 0) {
            while (top_candidates->Size() > inner_search_param.range_search_limit_size) {
                top_candidates->Pop();
            }
        }
        while (not top_candidates->Empty() &&
               top_candidates->Top().first > inner_search_param.radius + THRESHOLD_ERROR) {
            top_candidates->Pop();
        }
    }
    return top_candidates;
}

template <InnerSearchMode mode>
DistHeapPtr
ParallelSearcher::parallel_search_impl(const GraphInterfacePtr& graph,
                                        const FlattenInterfacePtr& flatten,
                                        const VisitedListPtr& vl,
                                        const void* query,
                                        const InnerSearchParam& inner_search_param,
                                        const LabelTablePtr& label_table) const {
    // std::cout << "----------------并行搜索器----------------" << std::endl;
    // auto t1 = std::chrono::high_resolution_clock::now();
    Allocator* alloc =
        inner_search_param.search_alloc == nullptr ? allocator_ : inner_search_param.search_alloc;
    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto computer = flatten->FactoryComputer(query);

    auto is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;
    uint64_t num_threads = inner_search_param.parallel_search_thread_count_per_query;

    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    
    float dist_ep = 0.0F;
    uint32_t count_no_visited_ep = 0;
    Vector<InnerIdType> to_be_visited_rid_ep(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> to_be_visited_id_ep(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> neighbors_ep(graph->MaximumDegree(), alloc);
    Vector<uint32_t> tasks_per_thread(num_threads,alloc);
    Vector<uint32_t> start_index(num_threads, alloc);
    
    Filter* attr_ft = nullptr;
    if (not inner_search_param.executors.empty() and inner_search_param.executors[0] != nullptr) {
        inner_search_param.executors[0]->Clear();
        attr_ft = inner_search_param.executors[0]->Run();
    }

    auto check_func = [&is_id_allowed, &attr_ft](InnerIdType id) {
        return (is_id_allowed == nullptr or is_id_allowed->CheckValid(id)) and
               (attr_ft == nullptr or attr_ft->CheckValid(id));
    };
    
    flatten->Query(&dist_ep, computer, &ep, 1, alloc);
    
    if (check_func(ep)) {
        top_candidates->Push(dist_ep, ep);
    }
    
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (dist_ep > inner_search_param.radius and not top_candidates->Empty()) {
            top_candidates->Pop();
        }
    }
    
    if (dist_ep < THRESHOLD_ERROR) {
        inner_search_param.duplicate_id = ep;
    }
    
    vl->Set(ep);
    //auto t2 = std::chrono::high_resolution_clock::now();
    hops++;
    std::pair<float, uint64_t> current_node_pair = {-dist_ep, ep};

    count_no_visited_ep = parallel_visit(graph,
                            vl,
                            current_node_pair,
                            inner_search_param.is_inner_id_allowed,
                            inner_search_param.skip_ratio,
                            to_be_visited_rid_ep,
                            to_be_visited_id_ep,
                            neighbors_ep);

    dist_cmp += count_no_visited_ep;

    uint32_t base = 0;
    uint32_t remainder = 0;
    uint32_t current_start = 0;

    if (num_threads) {
        base = count_no_visited_ep / num_threads;
        remainder = count_no_visited_ep % num_threads;
    }

    for (uint64_t i = 0; i < num_threads; ++i) {
        tasks_per_thread[i] = base + (i < remainder ? 1 : 0);
        start_index[i] = current_start;
        current_start += tasks_per_thread[i];
    }
    
    auto sub_search = [&](uint64_t thread_i) -> DistHeapPtr {
        //auto l1 = std::chrono::high_resolution_clock::now();
        auto sub_top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
        auto sub_candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);
        uint32_t sub_hops = 0;
        float dist = 0.0F;
        auto lower_bound = dist_ep;
        uint32_t count_no_visited = 0;
        Vector<InnerIdType> to_be_visited_rid(graph->MaximumDegree(), alloc);
        Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), alloc);
        Vector<InnerIdType> neighbors(graph->MaximumDegree(), alloc);
        Vector<float> line_dists(graph->MaximumDegree(), alloc);

        uint32_t start_index_offset = start_index[thread_i];
        uint32_t tasks_per_thread_offset = tasks_per_thread[thread_i];
        //auto l2 = std::chrono::high_resolution_clock::now();
        flatten->Query(line_dists.data(),
                           computer,
                           to_be_visited_id_ep.data() + start_index_offset,
                           tasks_per_thread_offset,
                           alloc);

        //auto l3 = std::chrono::high_resolution_clock::now();
        for (uint32_t i = start_index_offset; i < tasks_per_thread_offset + start_index_offset; i++) {
            dist = line_dists[i - start_index_offset];
            if (dist < THRESHOLD_ERROR) {
                inner_search_param.duplicate_id = to_be_visited_id_ep[i];
            }
            if (sub_top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                sub_candidate_set->Push(-dist, to_be_visited_id_ep[i]);
                //                flatten->Prefetch(sub_candidate_set->Top().second);
                if (check_func(to_be_visited_id_ep[i])) {
                    sub_top_candidates->Push(dist, to_be_visited_id_ep[i]);
                }
                if (inner_search_param.consider_duplicate and label_table != nullptr and
                    label_table->CompressDuplicateData()) {
                    const auto& duplicate_ids = label_table->GetDuplicateId(to_be_visited_id_ep[i]);
                    for (const auto& item : duplicate_ids) {
                        if (check_func(item)) {
                            sub_top_candidates->Push(dist, item);
                        }
                    }
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (sub_top_candidates->Size() > ef) {
                        sub_top_candidates->Pop();
                    }
                }

                if (not sub_top_candidates->Empty()) {
                    lower_bound = sub_top_candidates->Top().first;
                }
            }
        }
        // auto l4 = std::chrono::high_resolution_clock::now();
        // std::chrono::microseconds::rep dd1 = 0;
        // std::chrono::microseconds::rep dd2 = 0;
        // std::chrono::microseconds::rep dd3 = 0;
        // std::chrono::microseconds::rep dd4 = 0;
        // std::chrono::microseconds::rep dd5 = 0;

        while (not sub_candidate_set->Empty()) {
            //auto ll1 = std::chrono::high_resolution_clock::now();
            sub_hops++;
            auto current_node_pair = sub_candidate_set->Top();

            if (inner_search_param.time_cost != nullptr and
                inner_search_param.time_cost->CheckOvertime()) {
                break;
            }

            if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
                if ((-current_node_pair.first) > lower_bound && sub_top_candidates->Size() == ef) {
                    break;
                }
            }
            sub_candidate_set->Pop();
            //auto ll2 = std::chrono::high_resolution_clock::now();
            if (not sub_candidate_set->Empty()) {
                graph->Prefetch(sub_candidate_set->Top().second, 0);
            }
            //auto ll3 = std::chrono::high_resolution_clock::now();
            count_no_visited = parallel_visit(graph,
                                    vl,
                                    current_node_pair,
                                    inner_search_param.is_inner_id_allowed,
                                    inner_search_param.skip_ratio,
                                    to_be_visited_rid,
                                    to_be_visited_id,
                                    neighbors);

            //dist_cmp += count_no_visited;
            //auto ll4 = std::chrono::high_resolution_clock::now();
            flatten->Query(
                line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, alloc);
            //auto ll5 = std::chrono::high_resolution_clock::now();
            for (uint32_t i = 0; i < count_no_visited; i++) {
                dist = line_dists[i];
                if (dist < THRESHOLD_ERROR) {
                    inner_search_param.duplicate_id = to_be_visited_id[i];
                }
                if (sub_top_candidates->Size() < ef || lower_bound > dist ||
                    (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                    sub_candidate_set->Push(-dist, to_be_visited_id[i]);
                    //                flatten->Prefetch(sub_candidate_set->Top().second);
                    if (check_func(to_be_visited_id[i])) {
                        sub_top_candidates->Push(dist, to_be_visited_id[i]);
                    }
                    if (inner_search_param.consider_duplicate and label_table != nullptr and
                        label_table->CompressDuplicateData()) {
                        const auto& duplicate_ids = label_table->GetDuplicateId(to_be_visited_id[i]);
                        for (const auto& item : duplicate_ids) {
                            if (check_func(item)) {
                                sub_top_candidates->Push(dist, item);
                            }
                        }
                    }

                    if constexpr (mode == KNN_SEARCH) {
                        if (sub_top_candidates->Size() > ef) {
                            sub_top_candidates->Pop();
                        }
                    }

                    if (not sub_top_candidates->Empty()) {
                        lower_bound = sub_top_candidates->Top().first;
                    }
                }
            }
            // auto ll6 = std::chrono::high_resolution_clock::now();
            // dd1 += std::chrono::duration_cast<std::chrono::microseconds>(ll2 - ll1).count();
            // dd2 += std::chrono::duration_cast<std::chrono::microseconds>(ll3 - ll2).count();
            // dd3 += std::chrono::duration_cast<std::chrono::microseconds>(ll4 - ll3).count();
            // dd4 += std::chrono::duration_cast<std::chrono::microseconds>(ll5 - ll4).count();
            // dd5 += std::chrono::duration_cast<std::chrono::microseconds>(ll6 - ll5).count();
            
        }

        //std::cout << "while循环内部:" << dd1 << " " << dd2 << " " << dd3 << " " << dd4 << " " << dd5 << std::endl;

        if constexpr (mode == KNN_SEARCH) {
            while (sub_top_candidates->Size() > inner_search_param.topk) {
                sub_top_candidates->Pop();
            }
        } else if constexpr (mode == RANGE_SEARCH) {
            if (inner_search_param.range_search_limit_size > 0) {
                while (sub_top_candidates->Size() > inner_search_param.range_search_limit_size) {
                    sub_top_candidates->Pop();
                }
            }
            while (not sub_top_candidates->Empty() &&
                sub_top_candidates->Top().first > inner_search_param.radius + THRESHOLD_ERROR) {
                sub_top_candidates->Pop();
            }
        }

        // auto l5 = std::chrono::high_resolution_clock::now();
        // auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(l2 - l1).count();
        // auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(l3 - l2).count();
        // auto dur3 = std::chrono::duration_cast<std::chrono::microseconds>(l4 - l3).count();
        // auto dur4 = std::chrono::duration_cast<std::chrono::microseconds>(l5 - l4).count();
        // std::cout << "lambda内部：" << dur1 << " " << dur2 << " " << dur3 << " " << dur4 << std::endl;

        return sub_top_candidates;
    
    };             
    
    std::vector<std::future<DistHeapPtr>> futures;

    //auto t3 = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < num_threads; i++) {
        futures.emplace_back(pool->GeneralEnqueue(sub_search, i));
    }
    //auto t4 = std::chrono::high_resolution_clock::now();
    for (auto& f : futures) {
        auto result = f.get();
        while(not result->Empty()){
            top_candidates->Push(result->Top().first, result->Top().second);
            result->Pop();
        }
    }
    //std::cout << "1:" << top_candidates->Size() << std::endl;
    
    //auto t5 = std::chrono::high_resolution_clock::now();
    if constexpr (mode == KNN_SEARCH) {
        while (top_candidates->Size() > inner_search_param.topk) {
            top_candidates->Pop();
        }
    } else if constexpr (mode == RANGE_SEARCH) {
        if (inner_search_param.range_search_limit_size > 0) {
            while (top_candidates->Size() > inner_search_param.range_search_limit_size) {
                top_candidates->Pop();
            }
        }
        while (not top_candidates->Empty() &&
            top_candidates->Top().first > inner_search_param.radius + THRESHOLD_ERROR) {
            top_candidates->Pop();
        }
    }
    
    // auto t6 = std::chrono::high_resolution_clock::now();
    // auto dur1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    // auto dur2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    // auto dur3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    // auto dur4 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    // auto dur5 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();

    // std::cout << "主线程：" << dur1 << " " << dur2 << " " << dur3 << " " << dur4 << " " << dur5 << std::endl;

    return top_candidates;
}

void
ParallelSearcher::SetMutexArray(MutexArrayPtr new_mutex_array) {
    mutex_array_.reset();
    mutex_array_ = std::move(new_mutex_array);
}

}  // namespace vsag
