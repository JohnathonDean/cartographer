/*
 * Copyright 2016 The Cartographer Authors
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
#include "cartographer/mapping/2d/probability_grid.h"

#include <limits>

#include "absl/memory/memory.h"
#include "cartographer/mapping/probability_values.h"
#include "cartographer/mapping/submaps.h"

namespace cartographer {
namespace mapping {

ProbabilityGrid::ProbabilityGrid(const MapLimits& limits,
                                 ValueConversionTables* conversion_tables)
    : Grid2D(limits, kMinCorrespondenceCost, kMaxCorrespondenceCost,
             conversion_tables),
      conversion_tables_(conversion_tables) {}

ProbabilityGrid::ProbabilityGrid(const proto::Grid2D& proto,
                                 ValueConversionTables* conversion_tables)
    : Grid2D(proto, conversion_tables), conversion_tables_(conversion_tables) {
  CHECK(proto.has_probability_grid_2d());
}

// Sets the probability of the cell at 'cell_index' to the given
// 'probability'. Only allowed if the cell was unknown before.
// 将 索引 处单元格的概率设置为给定的概率, 仅当单元格之前处于未知状态时才允许
void ProbabilityGrid::SetProbability(const Eigen::Array2i& cell_index,
                                     const float probability) {
  // 获取对应栅格的引用
  uint16& cell =
      (*mutable_correspondence_cost_cells())[ToFlatIndex(cell_index)];
  CHECK_EQ(cell, kUnknownProbabilityValue);
  // 为栅格赋值 value
  cell =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(probability));
  // 更新bounding_box
  mutable_known_cells_box()->extend(cell_index.matrix());
}

// Applies the 'odds' specified when calling ComputeLookupTableToApplyOdds()
// to the probability of the cell at 'cell_index' if the cell has not already
// been updated. Multiple updates of the same cell will be ignored until
// FinishUpdate() is called. Returns true if the cell was updated.
//
// If this is the first call to ApplyOdds() for the specified cell, its value
// will be set to probability corresponding to 'odds'.
// 使用查找表对指定栅格进行栅格值的更新
bool ProbabilityGrid::ApplyLookupTable(const Eigen::Array2i& cell_index,
                                       const std::vector<uint16>& table) {
  DCHECK_EQ(table.size(), kUpdateMarker);  // 用于更新的表的大小是固定的
  // 把二维 cell_index 变换成一维索引 flat_index
  const int flat_index = ToFlatIndex(cell_index);
  // 获取对应栅格的原来的correspondence_cost值
  uint16* cell = &(*mutable_correspondence_cost_cells())[flat_index];
  // cell初始值为0(kUnknownCorrespondenceValue)，被更新一次之后会大于kUpdateMarker
  // 目的是防止在一次更新过程中多次更新一个点的值
  if (*cell >= kUpdateMarker) {
    return false;
  }
  // 那什么时候<kUpdateMaker？——更新完此帧点云后，见void Grid2D::FinishUpdate()
  // 更新了的index记录下，方便FinishUpdate一次性减去kUpdateMarker。
  mutable_update_indices()->push_back(flat_index);
  // 更新栅格值
  *cell = table[*cell];
  DCHECK_GE(*cell, kUpdateMarker);
  // 更新bounding_box
  mutable_known_cells_box()->extend(cell_index.matrix());
  return true;
}

GridType ProbabilityGrid::GetGridType() const {
  return GridType::PROBABILITY_GRID;
}

// Returns the probability of the cell with 'cell_index'.
// 获取 索引 处单元格的占用概率
//其上的 kValueToProbability 与 kValueToCorrespondenceCost 是两个转换表，第一个表示转换成被占用的概率，第二个表示未被占用的概率。
float ProbabilityGrid::GetProbability(const Eigen::Array2i& cell_index) const {
  if (!limits().Contains(cell_index)) return kMinProbability;
  return CorrespondenceCostToProbability(ValueToCorrespondenceCost(
      correspondence_cost_cells()[ToFlatIndex(cell_index)]));
}

proto::Grid2D ProbabilityGrid::ToProto() const {
  proto::Grid2D result;
  result = Grid2D::ToProto();
  result.mutable_probability_grid_2d();
  return result;
}

// 根据bounding_box对栅格地图进行裁剪到正好包含点云
std::unique_ptr<Grid2D> ProbabilityGrid::ComputeCroppedGrid() const {
  Eigen::Array2i offset;
  CellLimits cell_limits;
  // 根据bounding_box对栅格地图进行裁剪
  ComputeCroppedLimits(&offset, &cell_limits);
  const double resolution = limits().resolution();
  // 重新计算最大值坐标
  const Eigen::Vector2d max =
      limits().max() - resolution * Eigen::Vector2d(offset.y(), offset.x());
  // 重新定义概率栅格地图的大小
  std::unique_ptr<ProbabilityGrid> cropped_grid =
      absl::make_unique<ProbabilityGrid>(
          MapLimits(resolution, max, cell_limits), conversion_tables_);
  // 给新栅格地图赋值
  for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
    if (!IsKnown(xy_index + offset)) continue;
    cropped_grid->SetProbability(xy_index, GetProbability(xy_index + offset));
  }
  // 返回新地图的指针
  return std::unique_ptr<Grid2D>(cropped_grid.release());
}

// 获取压缩后的地图栅格数据
bool ProbabilityGrid::DrawToSubmapTexture(
    proto::SubmapQuery::Response::SubmapTexture* const texture,
    transform::Rigid3d local_pose) const {
  Eigen::Array2i offset;
  CellLimits cell_limits;
  // 根据bounding_box对栅格地图进行裁剪
  ComputeCroppedLimits(&offset, &cell_limits);

  std::string cells;
  // 遍历地图, 将栅格数据存入cells
  for (const Eigen::Array2i& xy_index : XYIndexRangeIterator(cell_limits)) {
    if (!IsKnown(xy_index + offset)) {
      cells.push_back(0 /* unknown log odds value */);
      cells.push_back(0 /* alpha */);
      continue;
    }
    // We would like to add 'delta' but this is not possible using a value and
    // alpha. We use premultiplied alpha, so when 'delta' is positive we can
    // add it by setting 'alpha' to zero. If it is negative, we set 'value' to
    // zero, and use 'alpha' to subtract. This is only correct when the pixel
    // is currently white, so walls will look too gray. This should be hard to
    // detect visually for the user, though.
    // delta处于[-127, 127]
    const int delta =
        128 - ProbabilityToLogOddsInteger(GetProbability(xy_index + offset));
    const uint8 alpha = delta > 0 ? 0 : -delta;
    const uint8 value = delta > 0 ? delta : 0;
    // 存数据时存了2个值, 一个是栅格值value, 另一个是alpha透明度
    cells.push_back(value);
    cells.push_back((value || alpha) ? alpha : 1);
  }
  // 保存地图栅格数据时进行压缩
  common::FastGzipString(cells, texture->mutable_cells());
  // 填充地图描述信息
  texture->set_width(cell_limits.num_x_cells);
  texture->set_height(cell_limits.num_y_cells);
  const double resolution = limits().resolution();
  texture->set_resolution(resolution);
  const double max_x = limits().max().x() - resolution * offset.y();
  const double max_y = limits().max().y() - resolution * offset.x();
  *texture->mutable_slice_pose() = transform::ToProto(
      local_pose.inverse() *
      transform::Rigid3d::Translation(Eigen::Vector3d(max_x, max_y, 0.)));

  return true;
}

}  // namespace mapping
}  // namespace cartographer
