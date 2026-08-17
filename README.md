# Ray Clock Probe

这个 MVP 用一条命令完成 Ray 集群的相对时钟建模：

- Ray Head 仍然是正常的 Ray worker，不预留为专用时间服务器。
- 一个轻量 actor 被固定到每台存活的物理 Ray 节点。
- Head 上的 actor 同时充当 UDP Clock Reference。
- 其他 `N-1` 个 actor 并行执行四时间戳探测，并在各自节点上拟合
  30 秒 piecewise affine model。
- driver 汇总所有模型为一个 `clock-session.json`。

## 前提

1. 集群已经由 Ray 启动，所有节点在同一 Ray cluster 中。
2. 节点安装 `iproute2` 和 `ethtool`。
3. 至少存在一条所有 worker 都能到达、且两端网卡都支持 Linux software
   timestamping 的 IPv4 路径。

检查网卡能力：

```bash
ethtool -T enp4s0f0np0
```

输出中应包含 `software-transmit` 和 `software-receive`。

## 安装

在已有 vLLM/Ray 环境中：

```bash
python -m pip install -e .
```

如果当前环境没有 Ray：

```bash
python -m pip install -e '.[ray]'
```

## 手动跟随 vLLM Profile 启停

在 vLLM 的 Ray Head 容器中，profile 开始前执行：

```bash
python -m clock_probe start
```

该命令创建 detached Ray actor 后立即退出，采样会继续在后台运行。可以查看状态：

```bash
python -m clock_probe status
```

然后正常调用 vLLM：

```bash
curl -X POST http://127.0.0.1:8000/start_profile
# 运行 workload；建议至少持续 30 秒
curl -X POST http://127.0.0.1:8000/stop_profile
```

最后停止校准并生成模型：

```bash
python -m clock_probe stop --output clock-session.json
```

同一个 Ray 集群只允许一个 active Clock Probe session。`stop` 完成后会清理
detached coordinator 和所有每节点 agent。

如果需要原来的一次性固定时长模式：

```bash
python -m clock_probe run \
  --duration-seconds 120 \
  --output clock-session.json
```

程序通过 `ray.nodes()` 自动发现物理节点，并使用
`node:__internal_head__` 资源识别唯一 Head。每个 actor 使用硬
`NodeAffinitySchedulingStrategy`，因此不会把两个节点的探测任务调度到同一台机器。

网卡自动选择会：

1. 在 Head 上运行 `ip -j address` 和 `ethtool -T`；
2. 只保留同时具有 `software-transmit`、`software-receive` 的 UP IPv4 网卡；
3. 让每个 worker 对候选 Head 地址运行 `ip -j route get`；
4. 只选择所有 worker 的路由出口也支持 software timestamping 的候选；
5. 将 worker UDP socket 绑定到该路由给出的源地址。

Ray 的 `NodeManagerAddress` 仅作为优先候选，不满足能力要求时会自动尝试其他
Head 网卡。也可以要求指定网卡或地址：

```bash
clock-probe-ray start --reference-interface enp4s0f0np0
clock-probe-ray start --reference-host 10.67.91.123
```

默认会把当前工程目录作为 Ray runtime working directory 上传；`.rayignore`
会排除虚拟环境、Git 数据及实验结果。若所有节点已经安装本 package，可传
`--working-dir ''` 禁止上传。

## 输出

`clock-session.json` 包含：

- Head identity model（offset 恒为 0）；
- 每个非 Head 节点的 piecewise affine model；
- 每段的 offset、drift ppm、有效 CLOCK_MONOTONIC 范围；
- held-out validation p95/max error；
- RTT、coverage、uncertainty 和 PASS/FAIL；
- 节点失败信息。

原始 JSONL 默认保存在每个节点的：

```text
/tmp/clock-probe/<session-id>/<ray-node-id>.jsonl
```

对应模型也会缓存在该节点：

```text
/tmp/clock-probe/<session-id>/<ray-node-id>.clock-model.json
```

模型的 offset 方向是 `reference_minus_source`。因此 source 节点时间转换为
Head 时间时使用：

```text
head_timestamp = source_timestamp + predicted_offset
```

模型不会修改系统时钟。它只用于离线 Trace 时间轴转换。
