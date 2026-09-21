# 柔索悬链标定服务

量跨中（或任意位置）的弧垂，反演水平张力 H，再按这根 H 把整档悬链吐出来。
只做柔索悬链几何这一块，HTTP 对外提供**标定（弧垂→H→曲线）**与**正算（H→曲线）**。

## 数学模型

左支座取 `(0, 0)`，右支座取 `(L, h)`（h 为带符号高差，右高取正），
形状参数 `c = H / w`（w 为单位长度重量）。曲线统一走双曲余弦，不使用抛物近似：

- 设最低点（顶点）无因次偏移为 `s = asinh(h / (2c·sinh(L/2c)))`，记 `ℓ = L/c`，
  两端相对最低点的无因次距为 `a = ℓ/2 − s`、`b = ℓ/2 + s`；
- 最低点水平位置 `x0 = a·c`，右端高度由 `cosh b − cosh a = h/c` 严格钉死；
- 索长 `S = c·(sinh a + sinh b)`；
- 测点 `x = ξL` 的弧垂按**弦高 − 索高**定义，用无减法的乘积恒等式求值：

      sag/c = (h/c)·ξ − 2·sinh(ξℓ/2)·sinh(s − (1−ξ)ℓ/2)

  等高档跨中即闭式 `c·(cosh(L/2c) − 1)`。

**正算与反演走同一个无因次函数**：给定 H 正算出测点弧垂，再把该弧垂
当测量值送回反演，解出的 H 必须在钉死容差 `1e-9`（相对）内回到原值。
每次响应都带 `forward_inverse_closure` 字段作为这把验收尺。

### 两种失败（绝不硬画一条对不上测量的曲线）

- `supports_unreachable`（422）：给定 H 正算时，高差大到最低点落到档外、
  索够不着两个支座；响应附 `max_feasible_H`。判据
  `|h/c| ≤ 2 sinh²(L/2c)`。
- `sag_height_difference_conflict`（422）：反演时测量弧垂比可行下限还小。
  可行域边界对应最低点贴在较低支座，该临界态在测点上的弧垂就是下限
  `minimum_feasible_sag`；弧垂小于它时没有任何 H 能同时满足两端边界与测量。

## 模块分工

| 文件 | 职责 |
| --- | --- |
| `app/catenary.py` | 双曲正算（唯一曲线数学）、端点参数、索长、可达性、临界档距 |
| `app/invert.py` | 由实测弧垂二分反演 H；正算→反演闭合检查（容差 `CLOSURE_RTOL=1e-9`） |
| `app/storage.py` | 几何档本地 JSON 文件读写（每档一个文件，原子写） |
| `app/validation.py` | 入参检查（H/w/档距/弧垂须正、测量点与取样点不得越界等） |
| `app/errors.py` | 统一错误码与 HTTP 状态 |
| `app/service.py` | 结果序列化编排 |
| `app/app.py` | Flask 路由 |

几何档持久化跨距几何（档距、高差、w、档名）；标定当次完成，不另建流水库。

## HTTP 接口

所有请求/响应均为 JSON。错误体形如
`{"error": <错误码>, "message": <中文原因>, ...}`。

### 几何档

- `POST /spans`：登记一档。请求体
  `{"name":"S1","span":300,"height_difference":35,"weight_per_length":10}`；
  重名报 409，带 `"overwrite": true` 可覆盖。
- `GET /spans`：列档，几何参数全文返回。
- `GET /spans/<name>`：查单档。
- `DELETE /spans/<name>`：删除。

### 标定（量弧垂 → H → 整条曲线）

`POST /calibrate`

```json
{
  "span": 300, "height_difference": 35, "weight_per_length": 10,
  "measured_sag": 18.5, "measurement_x": 120,
  "sample_count": 21
}
```

几何可以换成已登记的档名，此时只补测量：

```json
{"span_name": "S1", "measured_sag": 18.5, "measurement_x": 120,
 "samples": [0, 75, 150, 225, 300]}
```

- `samples`：显式水平取样点（允许落在端点，超出 `[0, L]` 拒绝）；
  不给则按 `sample_count`（默认 21）均匀取样。
- 响应含：`H`、`c`、`lowest_point`、`cable_length`、`supports`、
  `geometry`、`measurement`（测量点实测/计算弧垂与残差）、
  `curve`（每个取样点的 `x/y/chord/sag`）、`forward_inverse_closure`。

### 正算（已知 H → 整条曲线）

`POST /forward`，字段同上，用 `H` 代替 `measured_sag/measurement_x`。

### 其他

- `GET /health`：`{"status":"ok"}`
- 非法入参 400（H/w/档距/弧垂非正、测量点越界、取样越界、档名为空等）；
  档名对不上 404；几何矛盾 422（见上）。

## 构建与运行（单容器）

```bash
docker build -t catenary-service .
docker run --rm -p 8000:8000 -v "$PWD/data:/data" catenary-service
```

镜像基于 `python:3.12-slim`，gunicorn 单进程多线程对外服务；
几何档默认写 `/data`（可用 `CATENARY_DATA_DIR` 改），非 root 用户运行。

本地不装容器时也可：

```bash
pip install -r requirements.txt
flask --app app.app run --port 8000          # 开发
gunicorn --bind 0.0.0.0:8000 app.app:app     # 本地起服务
```

## 测试

```bash
pip install pytest
python -m pytest -q
```

锁住的行为（`tests/`）：

1. 等高跨中弧垂对上闭式 `c(cosh(L/2c)−1)`，最低点落在跨中；
2. 正算再反演 H 在 `1e-9` 相对容差内闭合（等高/不等高/跨中/偏心测点）；
3. 高差加大，最低点偏向较低一端，不再报跨中；
4. 大垂度档上抛物近似 `wL²/(8H)` 系统偏短一成以上，服务仍走双曲；
5. H 非正被拒；
6. 档距（及 w）非正被拒；
7. 弧垂与高差矛盾时标定失败并说明原因，而不是硬画；
8. 给定 H 够不着两端时报 `supports_unreachable`；
9. 档名对不上、弧垂非正、取样越界等当场给原因；
10. 具名档登记/列档/点名标定/正算→标定 H 对上的完整 HTTP 流程。
