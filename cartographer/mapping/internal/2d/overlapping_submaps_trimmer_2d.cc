/*
 * Copyright 2018 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/2d/overlapping_submaps_trimmer_2d.h"

#include <algorithm>

#include "cartographer/mapping/2d/submap_2d.h"

namespace cartographer {
namespace mapping {
namespace {

class SubmapCoverageGrid2D {
 public:
  // Aliases for documentation only (no type-safety).
  using CellId = std::pair<int64 /* x cells */, int64 /* y cells */>;
  using StoredType = std::vector<std::pair<SubmapId, common::Time>>;

  SubmapCoverageGrid2D(const MapLimits& map_limits)
      : offset_(map_limits.max()), resolution_(map_limits.resolution()) {}

  void AddPoint(const Eigen::Vector2d& point, const SubmapId& submap_id,
                const common::Time& time) {
    CellId cell_id{common::RoundToInt64((offset_(0) - point(0)) / resolution_),
                   common::RoundToInt64((offset_(1) - point(1)) / resolution_)};
    cells_[cell_id].emplace_back(submap_id, time);
  }

  const std::map<CellId, StoredType>& cells() const { return cells_; }
  double resolution() const { return resolution_; }

 private:
 // 第一个submap的最大xy(对应的cell id为0，0)
  Eigen::Vector2d offset_;
  // 地图分辨率
  double resolution_;
  // 存放所有cell的id,及id对对应的所有观测（submap的id，及submap的时间，该时间为submap第一帧激光的时间）
  std::map<CellId, StoredType> cells_;
};

// Iterates over every cell in a submap, transforms the center of the cell to
// the global frame and then adds the submap id and the timestamp of the most
// recent range data insertion into the global grid.
// 将所有的submap中的所有栅格都投影到coverage_grid中
std::set<SubmapId> AddSubmapsToSubmapCoverageGrid2D(
    const std::map<SubmapId, common::Time>& submap_freshness,
    const MapById<SubmapId, PoseGraphInterface::SubmapData>& submap_data,
    SubmapCoverageGrid2D* coverage_grid) {
  std::set<SubmapId> all_submap_ids;

  for (const auto& submap : submap_data) {
    auto freshness = submap_freshness.find(submap.id);
    if (freshness == submap_freshness.end()) continue;
    if (!submap.data.submap->insertion_finished()) continue;
    all_submap_ids.insert(submap.id);
    // 获取submap网格
    const Grid2D& grid =
        *std::static_pointer_cast<const Submap2D>(submap.data.submap)->grid();
    // Iterate over every cell in a submap.
    // 裁剪submap，因为里面有未经裁剪的submap
    Eigen::Array2i offset;
    CellLimits cell_limits;
    grid.ComputeCroppedLimits(&offset, &cell_limits);
    if (cell_limits.num_x_cells == 0 || cell_limits.num_y_cells == 0) {
      LOG(WARNING) << "Empty grid found in submap ID = " << submap.id;
      continue;
    }
    // submap在global SLAM坐标系下的位姿
    const transform::Rigid3d& global_frame_from_submap_frame = submap.data.pose;
    // submap在Local SLAM坐标系下的位姿
    const transform::Rigid3d submap_frame_from_local_frame =
        submap.data.submap->local_pose().inverse();
    for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
      const Eigen::Array2i index = xy_index + offset;
      if (!grid.IsKnown(index)) continue;
      // 根据cell id，转换为3d坐标(在Local SLAM坐标系下的坐标)
      const transform::Rigid3d center_of_cell_in_local_frame =
          transform::Rigid3d::Translation(Eigen::Vector3d(
              grid.limits().max().x() - grid.limits().resolution() * (index.y() + 0.5),
              grid.limits().max().y() - grid.limits().resolution() * (index.x() + 0.5),
              0));
      // 点在global SLAM坐标系下的位姿
      const transform::Rigid2d center_of_cell_in_global_frame =
          transform::Project2D(global_frame_from_submap_frame * submap_frame_from_local_frame * center_of_cell_in_local_frame);
      // 在全局地图的coverage_grid对应的栅格中插入覆盖子图的id和更新时间
      coverage_grid->AddPoint(center_of_cell_in_global_frame.translation(), submap.id, freshness->second);
    }
  }
  return all_submap_ids;
}

// Uses intra-submap constraints and trajectory node timestamps to identify time
// of the last range data insertion to the submap.
// 将最后插入submap的一帧lidar数据的时间戳作为submap的更新时间
std::map<SubmapId, common::Time> ComputeSubmapFreshness(
    const MapById<SubmapId, PoseGraphInterface::SubmapData>& submap_data,
    const MapById<NodeId, TrajectoryNode>& trajectory_nodes,
    const std::vector<PoseGraphInterface::Constraint>& constraints) {
  std::map<SubmapId, common::Time> submap_freshness;

  // Find the node with the largest NodeId per SubmapId.
  // 根据约束，寻找submap内所有的Node_id
  std::map<SubmapId, NodeId> submap_to_latest_node;
  // 遍历所有的约束，定位过程中仅在当前的轨迹中计算有约束，加载的地图轨迹中不存在约束
  for (const PoseGraphInterface::Constraint& constraint : constraints) {
    // 只遍历sequence边
    if (constraint.tag != PoseGraphInterface::Constraint::INTRA_SUBMAP) {
      continue;
    }
    auto submap_to_node = submap_to_latest_node.find(constraint.submap_id);
    if (submap_to_node == submap_to_latest_node.end()) {
      submap_to_latest_node.insert(
          std::make_pair(constraint.submap_id, constraint.node_id));
      continue;
    }
    // 找到最新的node_id
    submap_to_node->second =
        std::max(submap_to_node->second, constraint.node_id);
  }

  // Find timestamp of every latest node.
  // 遍历所有的submap及其对应的最新的Node_id
  for (const auto& submap_id_to_node_id : submap_to_latest_node) {
    auto submap_data_item = submap_data.find(submap_id_to_node_id.first);
    if (submap_data_item == submap_data.end()) {
      LOG(WARNING) << "Intra-submap constraint between SubmapID = "
                   << submap_id_to_node_id.first << " and NodeID "
                   << submap_id_to_node_id.second << " is missing submap data";
      continue;
    }
    auto latest_node_id = trajectory_nodes.find(submap_id_to_node_id.second);
    if (latest_node_id == trajectory_nodes.end()) continue;
    // 将最新的Node的时间取出作为submap时间
    submap_freshness[submap_data_item->id] = latest_node_id->data.time();
  }
  return submap_freshness;
}

// Returns IDs of submaps that have less than 'min_covered_cells_count' cells
// not overlapped by at least 'fresh_submaps_count' submaps.
std::vector<SubmapId> FindSubmapIdsToTrim(
    const SubmapCoverageGrid2D& coverage_grid,
    const std::set<SubmapId>& all_submap_ids, uint16 fresh_submaps_count,
    uint16 min_covered_cells_count) {
  std::map<SubmapId, uint16> submap_to_covered_cells_count;
  for (const auto& cell : coverage_grid.cells()) {
    std::vector<std::pair<SubmapId, common::Time>> submaps_per_cell(
        cell.second);

    // In case there are several submaps covering the cell, only the freshest
    // submaps are kept.
    // 对于每个cell保留最新的fresh_submaps_count个观测，记录每个观测中相同submap_id的保留数量
    if (submaps_per_cell.size() > fresh_submaps_count) {
      // Sort by time in descending order.
      std::sort(submaps_per_cell.begin(), submaps_per_cell.end(),
                [](const std::pair<SubmapId, common::Time>& left,
                   const std::pair<SubmapId, common::Time>& right) {
                  return left.second > right.second;
                });
      submaps_per_cell.erase(submaps_per_cell.begin() + fresh_submaps_count,
                             submaps_per_cell.end());
    }
    for (const std::pair<SubmapId, common::Time>& submap : submaps_per_cell) {
      ++submap_to_covered_cells_count[submap.first];
    }
  }
  std::vector<SubmapId> submap_ids_to_keep;
  // 如果该submap被保留的数量大于预设值min_covered_cells_count，则保留该submap
  for (const auto& id_to_cells_count : submap_to_covered_cells_count) {
    if (id_to_cells_count.second < min_covered_cells_count) continue;
    submap_ids_to_keep.push_back(id_to_cells_count.first);
  }

  DCHECK(std::is_sorted(submap_ids_to_keep.begin(), submap_ids_to_keep.end()));
  std::vector<SubmapId> result;
  // 求集合的差集，得到待删除的所有submap_id
  std::set_difference(all_submap_ids.begin(), all_submap_ids.end(),
                      submap_ids_to_keep.begin(), submap_ids_to_keep.end(),
                      std::back_inserter(result));
  return result;
}

}  // namespace

void OverlappingSubmapsTrimmer2D::Trim(Trimmable* pose_graph) {
  // 获取优化后的所有submap数据
  const auto submap_data = pose_graph->GetOptimizedSubmapData();
  // 设置的参数相关：每满min_added_submaps_count_个新的submap，执行一次删除
  if (submap_data.size() - current_submap_count_ <= min_added_submaps_count_) {
    return;
  }
  // 获取第一个submap的网格信息
  const MapLimits first_submap_map_limits =
      std::static_pointer_cast<const Submap2D>(submap_data.begin()->data.submap)
          ->grid()
          ->limits();
  // 构造SubmapCoverageGrid2D类
  SubmapCoverageGrid2D coverage_grid(first_submap_map_limits);
  // 计算每个submap内最后一个node的时间
  const std::map<SubmapId, common::Time> submap_freshness =
      ComputeSubmapFreshness(submap_data, pose_graph->GetTrajectoryNodes(),
                             pose_graph->GetConstraints());
  // 填充coverage_grid
  const std::set<SubmapId> all_submap_ids = AddSubmapsToSubmapCoverageGrid2D(
      submap_freshness, submap_data, &coverage_grid);
  // 找到待删除的所有submap id
  const std::vector<SubmapId> submap_ids_to_remove = FindSubmapIdsToTrim(
      coverage_grid, all_submap_ids, fresh_submaps_count_,
      min_covered_area_ / common::Pow2(coverage_grid.resolution()));
  current_submap_count_ = submap_data.size() - submap_ids_to_remove.size();
  // 逐个删除submap
  for (const SubmapId& id : submap_ids_to_remove) {
    pose_graph->TrimSubmap(id);
  }
}

}  // namespace mapping
}  // namespace cartographer

