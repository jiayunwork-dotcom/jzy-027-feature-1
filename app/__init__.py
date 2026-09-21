"""柔索悬链标定服务。

模块分工：
- catenary.py：双曲正算（唯一曲线数学）
- invert.py：由实测弧垂反演 H + 正算反演闭合
- storage.py：几何档本地文件读写
- validation.py：入参检查
- service.py / app.py：HTTP 编排
"""
