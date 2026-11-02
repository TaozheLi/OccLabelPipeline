# OCC Label Pipeline

[English](README.md) | **简体中文**

独立的 LiDAR 主导语义占据标签生成项目。原始点云先做语义分割，静态点通过 SLAM 融合，动态目标独立检测、跟踪、补全，并在最后合并，避免静态 KNN 修改车辆、行人等类别。

## 流程

```text
原始 LiDAR → 方向/强度字段适配 → KISS-ICP 位姿
          → PTv3 语义分割
          → TransFusion-L 检测、MCTrack 连续跟踪

静态分支：排除动态目标点 → 世界融合 → 可选人工修正
         → KNN → 孤立清理 → 地面补洞 → 未知点 KNN → 投影回各帧
动态分支：每100帧运动补偿融合 → BPA表面补全 → 车辆凸包内部补全

动态类最后合并 → 体素化 → 真实测量射线 free → ego 排除 → OCC
```

停车和低速目标仍在动态语义分支，速度只影响融合方式。动态合并之后不再做语义 KNN 或静态清理。

图片不参与语义标签生成或 SLAM。当前发布版不计算相机可见性，`mask_camera` 全 false，表示不可用；可视化可选使用六路图片。

## 安装与使用

测试环境为 **Python 3.10、NVIDIA CUDA GPU**。先阅读 [环境安装](docs/INSTALL.md)，安装 Pointcept、OpenPCDet 及其 CUDA 扩展，准备 MCTrack 和模型权重。仅安装 requirements.txt 不等于安装好模型框架。

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
cp configs/pcd.example.json configs/local.json
# NPZ/CSV录制格式使用 configs/npz_csv.example.json
```

修改配置里的原始数据目录、独立输出目录、三个框架路径、模型配置和权重，以及 LiDAR 高度、车体尺寸、方向和强度字段。相对路径以配置文件所在目录为基准。

```bash
occ-label validate --config configs/local.json
occ-label doctor --config configs/local.json
occ-label run --config configs/local.json
occ-label status --config configs/local.json
occ-label view --config configs/local.json
```

`run` 在前台运行；可用 `nohup` 放到后台。不会自动创建定时任务、网页服务或打开播放器。相同配置可重跑以复用已完成阶段；改权重或配置必须换新输出目录。连续跟踪中断后会重新运行跟踪，未完成的时序块会归档后重做。

## 输入

1. 逐帧目录：`000001/lidar.pcd`、`000001/meta.json`，后者包含秒单位 `lidar_time`。PCD 包含米单位 XYZ 和配置的强度字段。
2. NPZ/CSV：`lidar.csv` 包含 `ros_stamp_ns,file`；NPZ 包含 `xyz[N,3]` 和强度数组。可选 cameras.csv 包含 `camera,wall_time_ns,file`。

六路图片名称为 FL/F/FR/BL/B/BR；NPZ录制里的角色为 front left/front mid/front right/back left/back/back right。按时间最近匹配，记录时钟必须具有相同时间基准。缺相机不影响 LiDAR 生成；六路同步播放器只显示六图齐全的帧。

输入坐标需转为 **X前、Y左、Z上**。`yaw_to_vehicle_deg` 只旋转一次。通用格式不含逐点时间，因此本版关闭扫描内去畸变；可输入已去畸变扫描。`intensity_proxy_clip255` 是将强度裁剪到0..255的显式代理，不等同于真实反射率，可能产生域偏移。

## 输出与监督

最终标签在：

```text
<output_root>/occ_pipeline_v3_0/dynamic_last_v1/occ_full_interior_v1/<frame>/labels.npz
```

默认 0.4米体素、200×200×16网格、范围 `[-40,-40,-1,40,40,5.4]`；原始 LiDAR Z 加上配置的离地高度。ego框在平移后的网格坐标中定义。

有效监督通常为 `mask_lidar & ~mask_ego_excluded`。ID17配合有效mask表示已知free；ID17且mask无效表示未知，不应当作free参与loss。这里的mask_lidar包含保留的推断补全标签，不能理解为全部都是真实测量。详见 [标签定义](docs/LABELS.md)。

补洞、KNN、BPA和凸包都属于几何或语义推断，并非人工真值。车辆凸包可能跨越凹面并填充驾驶室空气；行人、自行车和摩托车不做实体内部填充。动态覆盖数为零仅验证合并规则，不代表检测精度为100%。

权重、原始数据和第三方框架不打包。需要用户自行提供兼容的nuScenes 16类PTv3配置/权重及TransFusion模型。默认支持可配置的数据路径和帧数，部分几何参数固定为当前验证方案。

[验证范围](docs/VERIFICATION.md) · [参与开发](CONTRIBUTING.md) · [第三方软件](THIRD_PARTY.md) · [许可证状态](LICENSE_STATUS.md)

## 可选人工静态修正

首次运行前将 `manual_review` 设为 true。静态世界地图完成后，流程以 `awaiting_manual_review` 状态停止，不会自动弹窗：

```bash
occ-label annotate --config configs/local.json
# 在原生Open3D工具中保存并关闭，然后继续：
occ-label run --config configs/local.json
```

包含多seed KNN、可旋转矩形、球选、俯视地面多边形和局部Z过滤。输入点云不覆盖，标注另存并保留备份；仅静态标签和擦除传入静态分支。下游完成后再改标注，需要新输出重新处理，不能复用旧结果。标注工具保留三维交互，最终OCC播放仍是固定二维俯瞰图。
