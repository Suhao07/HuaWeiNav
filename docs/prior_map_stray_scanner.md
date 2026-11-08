# Stray Scanner → 统一先验地图采集协议

状态：已实现导入器及格式/视频/坐标/竖正测试；已导入首份真实扫描 d655a5ce86（iPhone 17 Pro，App版本未记录），几何验收见 `archive/prior_map_capture_progress.md` 的 C 阶段记录。无卷尺真值，绝对尺度未验证。输入适配不等于机器人重定位或导航已完成。

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

- PNG 深度毫米转为 float32 米制 NPY，保留原始有效深度；导入不过滤低置信度，过滤由建图/布局显式参数决定。
- RGB 按packet顺序解码，与保存帧号、深度和位姿对应；以深度原生尺寸重采样 RGB。
- 视频时间戳：上游 `DatasetEncoder.swift` 以自增前的ARFrame计数作pts，首个保存帧pts=−1/60早于 `startSession(.zero)`，被MP4 edit list裁掉；默认解码少1帧且第k帧RGB配上第k+1帧深度/位姿。导入以 `OPENCV_FFMPEG_CAPTURE_OPTIONS=ignore_editlist;1` 解码全部packet。pts间隙来自编码积压时信号量丢帧，不影响packet顺序与保存帧一一对应。d655a5ce86：3447帧全部解出，25帧抽检RGB/深度边缘相关峰值均在偏移0。
- 竖正：手机竖握时ARKit图像x轴指向重力方向。按相机roll φ=atan2(R₂₀, −R₂₁)取全序列众数四分之一圈k=mode(round(−φ/90) mod 4)，RGB/深度/confidence同步顺时针旋转k次；每次u'=H−1−v, v'=u，K'=[[f_y,0,H−1−c_y],[0,f_x,c_x],[0,0,1]]，R'=R·[[0,1,0],[−1,0,0],[0,0,1]]，世界点集不变（d655a5ce86上5帧稠密点集最大差1.8e−15m）。manifest记录 `upright_quarter_turns_cw`。侧躺输入使检测器把键盘判为tv、反复出现ironing board，见C3。
- 逐帧 fx/fy/cx/cy 存在时直接读取（iPhone 17 Pro对焦使fx在1337–1360间变化）；旧版带表头数据使用 camera_matrix.csv，并显式记录为会话末帧内参来源，不能据此证明全程内参恒定。
- 内参缩放采用官方集成约定：`K_depth = diag(W_depth/W_rgb, H_depth/H_rgb, 1) @ K_rgb`。
- Confidence 随深度一同旋转后保存（不再逐字节复制）；缺失时写明 None 与实际数量，不伪造高置信度。共享建图器将其复制到证据包；`--min-depth-confidence` / `--min_depth_confidence` 显式开启过滤并写入receipt。
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
  --classes-file configs/prior_map/indoor_concepts.txt --min-depth-confidence 1

python scripts/build_capture_layout.py outputs/prior_capture/phone_scan \
  --output outputs/prior_diagnostics/phone_scan-layout/floorplan.json \
  --robot_height 1.5 --robot_radius 0.10 --min_depth_confidence 1
```

阈值1来自d655a5ce86的C4/C6对照：保留全部深度时conf0飞点造成对象重影并堵住桌间通道；只留conf=2则丢掉深色/反光物体并放开持续障碍格。布局、视点解析与建图须用同一阈值。

最后一条是共享建图入口示例，不代表该示例词表已通过真实场景质量验收。输入目录保留原样，输出目录须为新目录；转换失败写入 failed manifest 并抛出原错误。视频少帧、多帧或帧号与保存序列不一致均明确报错。

导入后的帧使用同一 observations.write_frame 协议。地图核心仍为 PriorMapData；confidence_ref 是可选证据字段，不引入手机专用地图类型。手持相机位姿只生成 observation_pose，不能直接作为机器人底盘目标。

## 验证边界

目前七项测试覆盖实际视频编码/解码、RGB 通道、毫米到米、逐帧内参、旧版内参、抽帧身份、四元数避免重复翻转、缺帧失败、LUT 明确拒绝、缺失 confidence，以及竖握扫描竖正后世界点集不变。

合成测试是工程契约；真实扫描的帧对齐、重力、相对尺度与回访漂移见C阶段记录。仍未验证：绝对尺度（无卷尺；LiDAR深度与ARKit VIO平移存在2–3%相对不一致）、机器人 map/odom 对齐、实物导航。manifest 的 physical_geometry_verified 固定为 false。
