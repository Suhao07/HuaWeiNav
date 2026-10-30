# Stray Scanner → 统一先验地图采集协议

状态：已实现导入器及格式/视频/坐标测试；尚无真实手机扫描的几何验收结果。输入适配不等于机器人重定位或导航已完成。

## 核对的上游版本

源码下载至项目工作空间的 `stray-scanner/`，与 STRIVE 仓库分开。

- 官方仓库：https://github.com/strayrobots/scanner
- 核对提交：`ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb`
- 格式：https://github.com/strayrobots/scanner/blob/ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb/docs/format.md
- 位姿编码：`StrayScanner/Helpers/OdometryEncoder.swift`
- 帧同步与视频：`DatasetEncoder.swift`、`VideoEncoder.swift`

上游的早期 `scripts/integrate.py` 读取旧版无表头 CSV/NPY 深度，不能直接当作当前格式读取器。当前适配依据实际 Swift 编码代码和格式文档，支持带表头的位姿 CSV。

## 坐标与标定

上游写入的四元数为：

\[
q_{WC}=q_{WA}\otimes q_{AC},\qquad q_{AC}=(1,0,0,0)_{xyzw}.
\]

其中相机光学轴转换已经发生。导入只变换世界坐标，不再次翻转相机轴：

\[
T^M_C=T^M_W T^W_C,\qquad
R^M_W=\begin{bmatrix}1&0&0\\0&0&-1\\0&1&0\end{bmatrix}.
\]

这与 Habitat 原生相机旋转的读取方式不同。两者最终都输出米制、右手 Z-up、光学相机 x 向右/y 向下/z 向前的 camera_to_map。

- PNG 深度毫米转为 float32 米制 NPY，保留原始有效深度，不自动过滤低置信度。
- RGB 顺序解码，与保存帧号、深度和位姿对应；以深度原生尺寸重采样 RGB。
- 逐帧 fx/fy/cx/cy 存在时直接读取；旧版带表头数据使用 camera_matrix.csv，并显式记录为会话末帧内参来源，不能据此证明全程内参恒定。
- 内参缩放采用官方集成约定：`K_depth = diag(W_depth/W_rgb, H_depth/H_rgb, 1) @ K_rgb`。
- Confidence 原文件保留；缺失时写明 None 与实际数量，不伪造高置信度。共享建图器将其复制到证据包，当前策略为 preserved_not_filtered。
- 带 distortion/*.bin 的输入目前明确报错，需要先完成畸变矫正；不会把未经核对的 LUT 当成无畸变模型使用。

## 使用

在 STRIVE 根目录，用有 OpenCV、NumPy、SciPy、Pillow 的环境执行：

```bash
python scripts/import_stray_capture.py /path/to/phone_scan \
  outputs/prior_capture/phone_scan --every 1

python scripts/diagnose_prior_capture.py \
  outputs/prior_capture/phone_scan --stride 16

bash docker/run_prior_capture_builder.sh \
  outputs/prior_capture/phone_scan outputs/prior_bundles/phone_scan \
  --classes chair table cabinet door window stairs lamp plant sofa bed toilet shelf
```

最后一条是共享建图入口示例，不代表该示例词表已通过真实场景质量验收。输入目录保留原样，输出目录须为新目录；转换失败写入 failed manifest 并抛出原错误。视频少帧、多帧或帧号与保存序列不一致均明确报错。

导入后的帧使用同一 observations.write_frame 协议。地图核心仍为 PriorMapData；confidence_ref 是可选证据字段，不引入手机专用地图类型。手持相机位姿只生成 observation_pose，不能直接作为机器人底盘目标。

## 验证边界

目前六项测试覆盖实际视频编码/解码、RGB 通道、毫米到米、逐帧内参、旧版内参、抽帧身份、四元数避免重复翻转、缺帧失败、LUT 明确拒绝和缺失 confidence。

这些是合成输入的工程契约测试，未验证手机的深度对齐、ARKit 漂移、内参随时间变化、真实跨视角重影、机器人 map/odom 对齐或实物导航。manifest 的 physical_geometry_verified 固定为 false，采集覆盖状态保留未知。
