# MiniMax、报警网页与拉回

## 1. 安全配置 MiniMax Key

真实 Key 只写入板端 `/etc/rdk-patrol.env`。推荐：

```bash
sudoedit /etc/rdk-patrol.env
```

填写：

```text
MINIMAX_API_KEY=<现场真实Key>
ROS_DOMAIN_ID=0
PYTHONUNBUFFERED=1
```

然后检查权限并重启：

```bash
sudo chown root:$(id -gn sunrise) /etc/rdk-patrol.env
sudo chmod 640 /etc/rdk-patrol.env
sudo systemctl restart rdk-patrol
```

不要：

- 把 Key 写入 `configs/system.yaml`、PowerShell、CMD 或截图。
- 在 shell 中执行会进入历史记录的 `export MINIMAX_API_KEY=<不要这样做>`。
- 把 `/etc/rdk-patrol.env` 加入拉回包。
- 在日志中打印请求头或完整 Key。

预检只判断 Key 是否非空，不显示内容。

## 2. 垃圾桶复核契约

触发条件是选择到垃圾桶点位并停留 5 秒。系统从该窗口选择一张清晰度、曝光和稳定性最好的**整幅巡检画面**，只上传这一张。

固定提示词：

> 图片中垃圾桶满了吗？如果图片中有多个垃圾桶或者一个垃圾桶有多个桶口，只要其中一个垃圾桶或者其中一个桶口满了就判断为已满，只输出一个词：已满 或 未满。

返回值经过去除首尾空白后仍必须严格等于“已满”或“未满”。解释、标点、JSON、英文或其他文本都视为无效响应，不触发“已满”报警。

MiniMax 网络失败、超时或无效响应时，请求连同关键帧元数据进入持久化重试队列；重启后继续处理。不得因重试而绕过同点位 5 分钟报警冷却。

## 3. 报警字段

车辆违停和垃圾桶已满：

```json
{
  "keyframe_image": "images/...",
  "event_name": "车辆违停",
  "beijing_time": "2026-07-27 14:30:00",
  "point_name": "禁停点1"
}
```

夜间人员逗留、夜间人群聚集和明火报警：

```json
{
  "keyframe_image": "images/...",
  "event_name": "明火报警",
  "beijing_time": "2026-07-27 14:30:00"
}
```

后三类必须完全省略 `point_name`。关键帧还应叠加检测框、事件证据和适用的距离信息。

## 4. 局域网只读网页

默认地址：

```text
http://192.168.66.65:8081/
```

功能：

- 每 5 秒读取最新报警。
- 按事件和点位筛选。
- 查看关键帧和四字段/三字段记录。
- 下载 JSONL。
- 下载自包含离线 HTML 台账。
- `GET /health` 健康检查。

网页 `/health` 返回完整相机、推理、引擎、录像和处理 FPS 健康快照，并强制附加 `read_only:true` 和 `alarm_count`。同一健康快照还会原子写入板端 `/var/lib/rdk-patrol/state/health.json`，服务短时不可访问时仍可用“查看RDK状态”脚本读取最近一次状态。

所有修改类 HTTP 方法都被拒绝，网页没有修改、删除、确认报警或机器人控制接口。首版没有账号密码，因此巡检网页的 8081 只应向可信局域网开放，不得暴露到互联网；8080 保留给相机网页。

常用只读地址：

```text
GET /
GET /health
GET /api/alarms/latest
GET /api/alarms?limit=200&event_name=车辆违停
GET /download/alarms.jsonl
GET /download/offline.html
```

## 5. Windows 拉回

双击：

```text
scripts\windows\拉回报警台账.cmd
```

或参数化运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Pull-RdkPatrol.ps1 `
  -BoardHost 192.168.66.65 `
  -User sunrise `
  -DestinationRoot D:\RDK巡检拉回
```

流程：

1. 板端运行 `rdk-patrol export-ledger`，生成自包含 `报警台账.html`。
2. 同时收集 JSONL、当前非密钥配置、systemd 状态、最近 24 小时日志和 SHA256。
3. 打成 `.tar.gz` 后复制到 Windows。
4. Windows 解压并直接打开 `报警台账.html`。

HTML 已把关键帧嵌入文件，断网电脑可双击查看和筛选。

拉回脚本不会删除：

- 板端报警记录。
- 板端报警关键帧。
- 板端生成的拉回包。
- 录像。

人工应先核对拉回包 SHA256、条数和关键帧，登记“已确认拉回”后，再按现场数据制度另行清理。首版故意不提供一键删除报警脚本，避免误删。

## 6. 钉钉后续接入

首版只预留关闭状态的 Outbox，不直接向钉钉发送。后续接入时应复用同一个固定报警 DTO，不改变当前网页/离线台账字段：

- 发送失败持久化重试。
- 使用幂等事件 ID 防止重复。
- 密钥仍放环境文件或受管密钥服务。
- 保留板端原始报警记录作为事实源。

接入钉钉前，必须单独确认机器人地址、签名方式、重试策略、网络安全和告警频率。
