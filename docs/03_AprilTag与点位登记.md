# AprilTag 打印、安装与车辆/垃圾桶点位登记

## 1. 点位规则

车辆违停和垃圾桶已满需要点位，另外三类事件不使用点位。默认预留：

| 点位 | 能力 | tag36h11 ID | 默认黑色方形边长 |
|---|---|---:|---:|
| `no_parking_01` | 车辆违停 | 101 | 0.18 m |
| `trash_01` | 垃圾桶复核 | 201 | 0.18 m |

同一巡检系统内，ID 必须全局唯一。新增点位前先维护一份 ID 台账，禁止复制同一 ID 到两个物理位置。

点位选择优先级固定为：

1. AprilTag 稳定识别成功：使用 Tag 对应点位。
2. Tag 未识别成功：尝试旧视觉指纹。
3. Tag 与视觉指纹冲突：Tag 获胜。
4. 两者均失败：跳过该点位的车辆/垃圾桶事件，不猜点位。

## 2. 生成和打印 tag36h11

在安装了 OpenCV ArUco 与 Pillow 的电脑或板端：

```bash
python3 scripts/tools/generate_apriltags.py \
  --ids 101 201 \
  --size-mm 180 \
  --quiet-zone-mm 10 \
  --dpi 300 \
  --output /var/lib/rdk-patrol/apriltags
```

打印要求：

- 选择“原始尺寸”或 100%，关闭“适合页面”“缩小到可打印区域”等缩放。
- 打印后测量**黑色标记外边长**，180 mm 标记建议误差不超过 1 mm。
- 四周至少保留 10 mm 连续白色静区，不写字、不打孔、不贴胶带。
- 使用哑光纸或哑光硬板，保持完全平整；不覆高反光膜。
- 对照 `打印说明.txt` 的 SHA256，确认打印文件未损坏或拿错 ID。

## 3. 安装位置

- 安装在机器人到点停留时，主检测视图能稳定看到的位置。
- 不被垃圾桶盖、车辆、人员或机械结构经常遮挡。
- 避开强背光、频闪灯、镜面和积水反射。
- 固定到刚性、静止结构，不能随垃圾桶盖开合或风吹摆动。
- 画面中标记边长建议不少于约 40 px，并尽量完整、清晰。
- 现场记录 Tag ID、点位名称、安装高度、实测边长和照片。

车辆地面禁停区与墙面 AprilTag 通常不共面，**不能只用 Tag 四角的平面单应性把墙面坐标直接当成地面 ROI**。正确做法是用 Tag 识别点位和相机姿态，再登记 Tag 坐标系到地面 ROI 的关系；若尚未完成三维登记，Tag 只负责点位选择，ROI 仍由经过几何验证的视觉指纹加载。

## 4. 登记前准备

1. 机器人准确停在日常巡检姿态，相机支架、分辨率和旋转与生产一致。
2. 通过导航预留接口发送“到点”和“暂停”；首版也可由人工停稳。
3. 保持至少 5 秒，确认 5 秒内 Tag 稳定识别不少于配置要求的连续帧数。
4. 备份现场配置：

```bash
cp -p /var/lib/rdk-patrol/site-config/points.yaml \
  /var/lib/rdk-patrol/site-config/points.yaml.before-registration
```

5. 确认配置中的 `family: tag36h11`、ID、实测 `size_m` 和实际打印一致。

必须先完成第 02 章双目标定。有效标定后，运行时会把主检测视图统一为**物理左目**，以保证明火框与视差坐标一致。查看：

```bash
grep -E '^(valid|physical_left_view|runtime_rotation):' \
  /var/lib/rdk-patrol/site-config/stereo_gs130w.yaml
```

确认 `valid: true`，再把输出的 `physical_left_view` 记为 `DETECTION_VIEW`，只能是 `top` 或 `bottom`：

```bash
# 按现场标定结果填写，不能照抄猜测。
DETECTION_VIEW=bottom
```

在点位停稳、Tag 清晰可见时，采集 6 张上下半视图参考图：

```bash
cd /opt/rdk-patrol/current
source /opt/tros/humble/setup.bash
python3 scripts/tools/collect_gs130w_pairs.py \
  --mode inspect \
  --headless \
  --count 6 \
  --auto-interval 1 \
  --topic /image_combine_jpeg \
  --layout vertical \
  --rotation ccw90 \
  --output /var/lib/rdk-patrol/registration/no_parking_01
```

登记时只使用生成的 `*_"$DETECTION_VIEW".png`。若物理左目是 `top`，就必须使用 `*_top.png`；不能沿用配置文件最初的 `bottom` 假设。

旧车辆 ROI/指纹来自旧主视图。若最终 `DETECTION_VIEW`、旋转、分辨率或相机姿态与旧配置不同，旧 legacy 锚点不再安全：先保留备份，再从现场 `points.yaml` 删除对应 `legacy.vehicle_config`/`legacy.trash_config` 引用，并以本次新采集的 `fingerprints` 作为 Tag 失败时的持久回退。绝不能跨镜头强行复用旧指纹。

登记工具至少接受 3 张图，并要求 Tag 角点最大标准差默认不超过 4 px；机器人未停稳会拒绝登记。

## 5. 车辆禁停点登记

车辆点位需要同时登记：

- `point_id` 和中文 `point_name`。
- AprilTag ID、实测黑色方形边长。
- 主视图中的地面禁停多边形 ROI。
- Tag 到地面 ROI 的三维关系；无法完成时，至少保留旧 ORB/视觉指纹和锚点几何。
- 正常停车姿态下的多帧视觉指纹。

在登记 ROI 时，沿真实禁停边界按顺序点击，不要自交；留出少量定位误差余量，但不要扩大到相邻车道。用无车画面登记后，再放入测试车辆验证 2 秒逗留阈值。

`configs/legacy/vehicle_site_rois.yaml` 已作为旧视觉指纹来源保留。它可能来自旧相机姿态，现场必须做遮挡 Tag 的回退测试，不能仅因文件存在就判定有效。

从刚采集的主视图图片中选择一张清晰无车图，交互绘制当前 ROI：

```bash
ls /var/lib/rdk-patrol/registration/no_parking_01/*_"$DETECTION_VIEW".png

# 从上面的列表复制一条真实绝对路径并粘贴。
read -r -p "参考图绝对路径: " SELECTED_IMAGE
test -f "$SELECTED_IMAGE" || { echo "文件不存在"; exit 1; }

python3 scripts/tools/draw_point_roi.py \
  --image "$SELECTED_IMAGE" \
  --roi-id no_parking_zone_01 \
  --output /var/lib/rdk-patrol/registration/no_parking_01_rois.yaml
```

左键加点、右键撤销、Enter 完成、Esc 取消。工具拒绝少于 3 点、越界、重复、自交或面积过小的多边形，并生成 `.preview.jpg` 供双人复核。纯 SSH 环境可在运维电脑运行同一工具，或使用可重复的 `--polygon '名称:x1,y1;x2,y2;x3,y3'`，但必须对照预览图复核。

### 首版推荐：Tag 选点 + ORB 几何加载 ROI

墙面 Tag 与地面禁停区不共面，而且当前没有现场三维测量文件时，必须使用安全的 `--identity-only`：

```bash
python3 scripts/tools/register_point.py \
  --points /var/lib/rdk-patrol/site-config/points.yaml \
  --point-id no_parking_01 \
  --images /var/lib/rdk-patrol/registration/no_parking_01/*_"$DETECTION_VIEW".png \
  --min-good-images 3 \
  --max-corner-std-px 4 \
  --rois-yaml /var/lib/rdk-patrol/registration/no_parking_01_rois.yaml \
  --identity-only
```

该模式会记录 Tag 角点、图像 SHA256 和新的 `fingerprints`，但保留 `tag.registration_required: true`：Tag 稳定识别后负责选择 `no_parking_01`，同点视觉指纹的 ORB/RANSAC 几何负责安全投影旧 ROI。指纹几何失败时返回空 ROI，不会把旧静态区域硬套到当前画面。这是“宁漏报、不误报”的首版推荐。

因此预检仍会提示该车辆点“AprilTag 现场登记未完成”。准确含义是“尚未完成 Tag 到地面 ROI 的直接三维空间登记”，不是 Tag 身份或视觉指纹没有登记。该警告必须写入验收记录；在没有可靠三维测量时，不得为了清除警告降低安全门。

### 高级模式：Tag 6DoF + 地面 ROI 三维登记

只有完成现场测量时，才创建例如：

```yaml
roi_points_tag_m:
  no_parking_zone_01:
    - [x1_m, y1_m, z1_m]
    - [x2_m, y2_m, z2_m]
    - [x3_m, y3_m, z3_m]
    - [x4_m, y4_m, z4_m]
```

上面是故意不可直接运行的字段模板，必须把变量替换为实测数字。坐标必须是以 Tag 为参考系、单位为米的真实三维坐标，不是像素或估算值。运行时会从已验证的 GS130W 标定自动注入物理左目在旋转后坐标系中的内参和畸变参数，所以本模式必须在双目标定 `valid:true` 后使用。由具备测量能力的人员复核后运行：

Tag 坐标系固定为：

- 原点：黑色方框中心。
- `+X`：沿印刷平面向右。
- `+Y`：沿印刷平面向上。
- `+Z`：从印刷正面指向观察者。
- 单位：米。
- 图像四角检测顺序：左上、右上、右下、左下（TL/TR/BR/BL）。

运行时会检查姿态重投影误差，并拒绝投影到相机后方或不满足正深度门控的 ROI。坐标轴、角点顺序或正负号不确定时必须停止登记，不能靠试值让画面看起来“差不多”。

```bash
python3 scripts/tools/register_point.py \
  --points /var/lib/rdk-patrol/site-config/points.yaml \
  --point-id no_parking_01 \
  --images /var/lib/rdk-patrol/registration/no_parking_01/*_"$DETECTION_VIEW".png \
  --rois-3d-yaml /var/lib/rdk-patrol/registration/no_parking_01_rois_3d.yaml
```

这会把 `registration_required` 设为 `false`。不得为了消除预检警告而伪造三维坐标。`--rois-coplanar-with-tag` 只允许 ROI 与 Tag 确实位于同一物理平面时使用，墙面 Tag + 地面停车区禁止使用。

## 6. 垃圾桶点登记

垃圾桶点位需要登记：

- `point_id`、中文 `point_name`。
- AprilTag ID 和实测 `size_m`。
- 到点停稳后的整幅主视图视觉指纹。
- 日间、夜间或灯光变化明显时的多个有效锚点。

垃圾桶复核不裁剪单个桶口上传：在 5 秒停留窗口内，从整幅巡检画面选择质量最好的一张发送 MiniMax。Tag 只用于选择正确点位，不应出现在报警字段以外替代点位名称。

旧 `configs/legacy/trash_site_points.yaml` 若没有有效 anchors，回退能力实际不可用，必须现场采集新视觉指纹。

采集垃圾桶点参考图：

```bash
python3 scripts/tools/collect_gs130w_pairs.py \
  --mode inspect \
  --headless \
  --count 6 \
  --auto-interval 1 \
  --topic /image_combine_jpeg \
  --layout vertical \
  --rotation ccw90 \
  --output /var/lib/rdk-patrol/registration/trash_01
```

垃圾桶点不需要敏感 ROI，可以完成普通身份登记：

```bash
python3 scripts/tools/register_point.py \
  --points /var/lib/rdk-patrol/site-config/points.yaml \
  --point-id trash_01 \
  --images /var/lib/rdk-patrol/registration/trash_01/*_"$DETECTION_VIEW".png \
  --min-good-images 3 \
  --max-corner-std-px 4
```

工具会把 `tag.registration_required` 设为 `false`，并写入最多 8 个新旧 `fingerprints`。Tag 识别失败时，运行时使用这些指纹回退。

## 7. 配置生效

最终登记文件必须写入：

```text
/var/lib/rdk-patrol/site-config/points.yaml
```

不要手工改 `registration_required`。登记工具根据 `identity-only`、三维 ROI 或共面条件决定是否可以安全清除此标志。典型登记后的嵌套结构为：

```yaml
points:
  - point_id: trash_01
    point_name: 垃圾桶点位1
    capabilities: [trash_review]
    tag:
      family: tag36h11
      id: 201
      size_m: 0.18
      registration_required: false
    reference_tag_corners: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    fingerprints:
      - {schema_version: rdk-patrol-visual-fingerprint/v1, ...}
    registration:
      accepted_images: 6
      max_corner_std_px: 1.2
      spatial_mode: identity_no_roi
```

应用：

```bash
sudo /opt/rdk-patrol/current/scripts/rdk/apply-site-config.sh
sudo -u sunrise /opt/rdk-patrol/current/scripts/rdk/preflight.sh
```

局域网页是只读报警页面，不能从网页修改点位、删除报警或控制机器人。

## 8. 必做故障测试

每个车辆和垃圾桶点位都记录以下结果：

| 场景 | 预期 |
|---|---|
| Tag 清晰，视觉指纹匹配 | 选择 Tag 对应点位 |
| Tag 清晰，视觉指纹错误指向另一点 | Tag 获胜 |
| 遮住 Tag，旧视觉指纹有效 | 自动加载旧指纹对应区域 |
| 遮住 Tag，视觉指纹也失败 | 跳过点位事件，不报警、不误用上一点 |
| 检出未知 Tag ID | 不映射到任何已知点位 |
| 机器人轻微停点偏差 | ROI 稳定投影，不能漂到相邻区域 |
| 相机明显偏位 | 点位定位失败或健康告警，不能强行沿用 ROI |

“Tag 未识别时使用旧视觉指纹”是运行时回退，不代表允许省略 AprilTag 现场安装和登记。
