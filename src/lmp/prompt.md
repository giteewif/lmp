
### tensor-index-json
1. 将 同层self_attn，gate 参数安排在同个设备，将不同层 self_attn, gate 分散到不同设备
2. 同层的专家，尽量均匀分配到设备
3. 其他参数都放在第一个设备
4. 参数获取都使用 mlpllm的接口，没有相应接口就创建