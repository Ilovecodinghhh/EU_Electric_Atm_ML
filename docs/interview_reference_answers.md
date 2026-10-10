# 欧洲日前电价预测项目：简短参考回答

本答案册与 [`interview_questions.md`](interview_questions.md) 的章节及题目顺序逐项对应。回答采用面试口径：先给结论，再给证据和边界。未完成的研究明确标注为“尚未验证”，避免把预测结果包装成交易结果。

## 一、30 秒项目介绍

1. 我构建了一个七国日前电价预测管线，在严格的 point-in-time 时间轴下比较价格历史、天气预报水平和 forecast revision。183 个完整 UTC 日的 OOS 中，Full Ridge 相对 persistence 将 MAE 降低 21.32%。
2. 它服务于拍卖前的价格预估，可用于报价、风险预算或进一步构造交易信号；当前项目交付的是预测研究，不是自动交易系统。
3. 股票收益与天气之间链条太长且噪声大，而天气对可再生出力和电价的机制更直接，因此我把目标移到更接近基本面的电价。
4. 核心假设是：决策时点已发布的天气预报水平及其更新，对下一交付日电价具有价格历史之外的增量信息。
5. Full Ridge MAE 为 17.83 EUR/MWh，比 persistence 低 4.83 EUR/MWh，即 21.32%；天气全特征相对 price-only 的增量为 2.91 EUR/MWh。
6. 当前证据不能证明 Ridge 或 HistGradient 任一方在预测上更优。HistGradient 的点估计更好；若必须确定主规格，我按简约性选择 Full Ridge，并把 HistGradient 保留为 challenger，因为它的额外复杂度尚未带来可靠增量。
7. 最大局限是只有 2024 年下半年 183 个完整 UTC 日的 OOS，尚未覆盖多个能源市场 regime，也没有可执行交易回测。
8. 可写为：“构建七国 point-in-time 日前电价预测管线；183 天 walk-forward OOS 中 Full Ridge 相对 persistence 降低 MAE 21.32%，并用 block bootstrap 验证增量。”
9. 正结果是天气预报水平对 price-only 有稳定增量；负结果是 revision 单独无效、增量很小，Norway 也没有战胜 persistence。
10. sklearn 只是实现工具，研究价值在时间戳纪律、数据 vintage、强基线、walk-forward、ablation 和依赖结构稳健的统计推断。

## 二、研究问题与因果机制

1. 预报天气影响风光出力和负荷，再改变 residual load、边际机组与跨区流量，最终影响日前市场出清价。
2. 风速影响风电供给，短波辐射影响光伏供给，气温影响采暖和制冷负荷；三者共同改变 residual load。
3. Revision 表示市场基本面预期的新增变化，理论上比静态天气水平更接近“新信息”。
4. 市场会吸收公开信息，所以我不预设存在 alpha；项目只检验公开预报能否改善一个简单统计基线，真正 alpha 还需对比市场共识与可交易价格。
5. 当前模型主要预测基本面条件下的价格水平，不足以证明能预测市场定价误差。
6. 不一定线性。风功率曲线、拥塞、负价和边际机组切换都可能造成阈值与交互，因此 Ridge 是透明基线而非最终结构假设。
7. 在拥塞、弃风、负价、风机切出或邻国出口受限时，更多风不一定继续压低本地价格。
8. 现有日聚合风速无法表达 cut-out 风险；应加入站点级小时风速、风机功率曲线和极端风指示变量。
9. 气温对负荷具有季节和国家异质性，冬季采暖、夏季制冷以及电气化程度都会改变斜率。
10. 辐射需要结合云量、太阳高度角和历史光伏装机映射为潜在发电，直接使用辐射只是粗代理。
11. 负价区间通常对应高可再生出力和低负荷，关系会更非线性，并受到补贴、约束和最小运行功率影响。
12. 燃料与碳价决定热电边际成本，是重要遗漏变量；当前结论应解释为天气在简化信息集中的增量，而非完整结构模型。
13. 水电、核电、检修与互联约束同时影响价格并可能与天气相关，应在下一版加入以减少遗漏变量偏差。
14. 我通过日历控制、price-only baseline、feature-family ablation 和跨 fold 稳定性做初步区分；严格因果识别尚未完成。
15. 它首先是预测研究。没有识别外生处理效应，也没有完整交易规则和执行成本，因此不能称为因果或交易研究。

## 三、目标变量与决策时点

1. 对每个国家，把目标交付日内可用的小时日前价格取均值，形成一个 country-day 的 EUR/MWh 目标。
2. 当前文件被当作国家日前价格代理，但仓库仍需补齐交易所和产品级 provenance；在补齐前我不会声称它是某个官方指数。
3. 目前代码没有把数据源追溯到具体交易所，这是数据治理缺口，下一步应记录 provider、market area、product 与下载时间。
4. 代码按 UTC 日 resample 后计算 mean；这简单但没有完整处理市场当地日和 DST，是已知限制。
5. 当前 UTC 聚合会自然得到不同小时数，但未显式验证 23/25 小时权重；生产版应按 bidding-zone local delivery day 聚合。
6. 日均目标先降低维度、验证天气是否有基本增量；代价是丢失峰谷、拥塞和极端小时信息。
7. 会。日均会平滑最有价值的尖峰，因此结果只能说明 baseload-like 日水平可预测性。
8. 下一步应预测 24 小时曲线或 peak/base 产品，因为它们更贴近拍卖报价和风险管理。
9. 冬季大致是中欧 10:00、爱尔兰 09:00；夏季当地钟点整体前移一小时。固定 UTC 并不等于固定市场本地时点。
10. 这是必须逐市场验证的关键假设；当前只使用保守代理时间，尚未以每个交易所规则做完整 gate-closure 审计。
11. 会。固定 UTC 在 DST 前后对应不同当地时钟时间，因此生产设计应以市场时区和 auction calendar 定义决策时点。
12. 目标日前价格通常在 D-1 拍卖出清后可知；模型必须在对应 gate closure 前冻结输入。
13. 会造成严重错位，甚至把结果发布时间当成交付时间。必须通过 provider schema 和少量人工样本双重验证。
14. `price_lag1` 是 D-1 的交付价，通常在 D-2 已出清，因此 D-1 09:00 可知；但该结论仍依赖原始数据确实是日前产品。
15. 不完全相同，节假日和周末可能有不同流动性与负荷结构；应加入市场日历并检查拍卖规则。
16. 价格水平直接对应业务单位且 persistence 是自然基线；预测变化可能更平稳，但解释与部署目标不同，应作为稳健性规格比较。
17. 是，所以必须与 persistence 和 seasonal naive 比较；高 R2 不能单独作为成功证据。
18. 这会把问题改写为预测“偏离昨日价格的部分”，更容易隔离新增信息，值得作为下一版主规格之一。
19. 目标不裁剪，保留真实经济尾部；输入用 train-only winsorization 控制单个极值对估计的影响。
20. 目前尚未完成。需要明确可交易产品、gate closure、报价曲线、头寸和结算后，预测才可转成执行决策。

## 四、国家与市场范围

1. 这七国来自现有节点和价格数据中能够形成完整 country series 的交集，不是声称最优的欧洲市场样本。
2. 主要由数据可得性决定，因此存在选择偏差；应在研究开始前冻结 universe 并记录排除原因。
3. 当前数据没有提供同等质量覆盖，不能把缺失国家解释成经济选择；扩展前应先解决数据一致性。
4. 会。可用数据更完整的国家可能恰好更成熟、更互联，结果未必能外推到整个欧洲。
5. 应将德国目标明确映射到相应年份的 DE-LU bidding zone，而不是泛称国家价格。
6. 不应简单汇总。NO1 至 NO5 受水文和拥塞影响显著不同，国家平均很可能是 Norway 失败的重要原因。
7. 应拆分 DK1/DK2，分别匹配天气、互联和价格；国家 dummy 无法恢复被平均掉的区域结构。
8. I-SEM 的市场设计和互联结构不同，统一模型只能作为初始基线，最终可能需要区域特定模型。
9. 不能。Dummy 只控制截距，不能表达不同天气斜率、非线性和市场规则。
10. 不等于。共同天气、燃料和跨境流量让同日国家误差相关，有效独立时间样本更接近 183 天。
11. 应报告 country-level 指标、宏观等权指标和 leave-one-country-out 结果，避免 pooled 数字掩盖集中度。
12. 统一模型可共享有限样本并降低方差；分国模型更灵活但样本很少，容易过拟合。
13. 是。层级模型可让国家斜率围绕共同分布收缩，在共享信息和异质性之间折中。
14. 当前只隐式通过价格历史吸收，未显式建模。可加入邻国天气、互联容量与 flow features。
15. 目标、天气和日历都要改为 bidding-zone 粒度，并重做 zone mapping、DST、互联和分层评估。
16. 应先扩展到官方 bidding zones，再决定是否仍需要国家级汇总作为高层 benchmark。

## 五、电价数据质量

1. 当前仓库保留了原始项目输入，但来源链条不够完整；这是必须补齐的审计项，不能在面试中虚构来源。
2. 原项目以风电场为节点，把国家级价格广播到节点；这对图模型制造了重复而非真实空间信息。
3. 代码按节点元数据映射国家，将同国可用列逐时取中位数，得到一条国家级序列。
4. 中位数对少量异常或不一致复制更稳健；理想状态下重复列应完全相同，并通过断言验证。
5. 它可能隐藏列映射错误、区域价差或数据污染，所以还应输出同国列间差异审计。
6. 当前没有系统报告，这是应新增的数据质量测试，例如逐国最大绝对差和不一致小时比例。
7. 聚合时忽略 NaN，后续特征由 train-only 中位数插补；目标缺失样本直接排除。
8. 连续缺失往往对应 provider 或市场故障，会形成 regime bias；随机缺失主要增加方差，二者不能同样处理。
9. 代码排序并统一为 UTC，但尚需显式检查重复、频率间断和 timezone provenance。
10. 尚未建立单位断言；应在 metadata 中固定 EUR/MWh，并对典型范围和单位转换做检查。
11. 当前默认输入已是 EUR/MWh，但没有可审计 FX 转换记录；非欧元市场必须补 point-in-time FX 或使用官方欧元口径。
12. 先与官方事件、邻国价格和原始源交叉验证；无法解释的孤立值才标记为数据问题，不能仅按阈值删除。
13. 输入极值会扭曲参数估计，目标极值却是要预测的真实结果；因此只对输入做 train-only 稳健处理。
14. 1%/99% 是保守起点，不是理论最优；应做 0.5/99.5、2/98 和不裁剪的敏感性分析。
15. 有这个风险，所以目标不裁剪，并应单独报告尖峰日误差，确保模型没有靠回避尾部获得好看均值。
16. 尚未完整量化。应逐 fold 比较均值、波动、负价率、尖峰率和 PSI/KS 等漂移指标。
17. 当前报告没有该统计，应补入 data audit；这是判断 regime 与损失函数的重要信息。
18. 尚未确认发布许可；公开仓库前应核对 provider license，必要时只提交下载脚本和 checksum。
19. Parquet 更适合保留类型、压缩和快速读取；CSV 可读性好，但应配 schema、checksum 与数据字典。
20. 我会把原始层设为不可变文件，增加 schema、时间覆盖、重复、单位、缺失、DST 和跨列一致性检查。

## 六、天气预报数据与 Forecast Revision

1. ERA5 是事后再分析，可能融合交付后观测；历史预报更接近当时真实可用的信息集。
2. 如果用 D 日 realized weather 预测 D 日价格，就把交付期间才观测到的信息放进了决策时点。
3. Open-Meteo 可能聚合不同 NWP 来源，具体 archive 的模型与版本需按官方元数据核实；当前代码没有锁定底层模式版本。
4. API 提供 lead-day 字段，但当前数据没有逐小时原始 issue timestamp，这是可用性证明中的不足。
5. 它们表示相对有效时刻早两天、早三天保存的预报版本；精确定义应以 provider 文档和样本核验为准。
6. D-1 更新可能在 09:00 UTC 尚未对所有交付小时完整可得；day-2/day-3 更保守，牺牲新鲜度换取时间戳安全。
7. 当前规则把最晚可用时间设为 D-2 23:00 UTC，确保早于 D-1 09:00；这是保守代理，不是逐 run 的发布时间事实。
8. 主要是根据 lead-day 语义构造的保守假设，尚未逐条验证 issue metadata，因此结论应明确这一边界。
9. 有可能。历史服务若用新模型重算旧预报，会产生 vintage bias；需确认 archive 是 immutable forecast vintage。
10. 会。模型升级会改变预报误差分布，应记录 model/version 并按升级日期做 regime 检查。
11. API 中所需变量在更早时期覆盖不足，所以有效实验从 2024 年 3 月开始；这不是经济上选择的起点。
12. 已知部分国家风速和辐射约有 2% 至 3% 缺失，但应在正式报告中逐国逐变量列出准确比例。
13. 尚未系统分解；应按国家、月份、lead 和变量生成 missingness heatmap，并检验是否与高价日相关。
14. 会。中位数可能把缺失解释为常态，因此应同时加入 missing indicator 并做完整样本稳健性分析。
15. 应加入。它既能防止插补值被误读，也能暴露数据供应中断本身是否与市场状态相关。
16. 绝对差保持原单位、易解释；相对差可跨季节比较但在接近零时不稳定，二者应作为并行规格。
17. 风速 revision 是 m/s，但发电影响近似非线性；应通过功率曲线、容量和 cut-in/rated/cut-out 区间转换。
18. 应使用。Ensemble spread 直接表达预报不确定性，可能比两个 deterministic vintage 的差更稳健。
19. Forecast error 可评估预报质量，但 realization 在决策时点不可得，不能作为实时输入。
20. 不可得，所以只能用于事后诊断、模型加权或训练标签，不能未经 lag 直接进入特征。
21. 三者分别代理风电、光伏和温度负荷，是最小机制集合；选择是务实起点，不是完整最优集。
22. 尚未加入云量、风向、降水、气压和湿度；应按明确机制和 out-of-sample ablation 逐步扩展。
23. Mean 表示全天水平，std 表示日内变化，min/max 表示极端条件；它们压缩了小时路径信息。
24. 对 mean、min/max 部分情况下关系不同；逐小时相减再聚合保留同一小时配对，通常更符合 revision 定义。
25. 不等价。当前代码先算 day-2 与 day-3 各自的日内 std，再相减，而不是计算逐小时 revision 的 std；这一特征应重命名或重构。
26. 它否定“revision 单独足够预测价格”的强假设，但不否定它作为价格历史条件下的小增量信息。
27. 统计上为正，但经济意义尚未证明；需结合报价敏感度、市场基准、成本和稳定性判断。
28. 尚未充分拆解。下一步要做按变量、国家、月份和 fold 的 leave-one-family-out 与系数稳定分析。
29. 应保存可合法缓存的 immutable raw vintages，并准备 ECMWF、DWD 或其他官方 archive 的替代适配器。

## 七、空间表示与风电场元数据

1. 它用一个请求点近似国家风电重心，降低 API 调用和维度，是 proof of concept，而非最终物理表示。
2. 损失很大，尤其是面积大、海陆并存或多天气区国家；它无法表达空间分散和局地锋面。
3. 可以。数学质心不保证是代表性气象位置，因此应检查坐标并改为站点级预测后加权。
4. 站点级方案更符合物理机制，但需要更多 API、历史容量和缺失处理；它是优先级很高的升级。
5. 不合理，因为天气场和功率曲线都非线性；正确顺序更接近“各站点天气转出力，再按容量汇总”。
6. 当前元数据未形成逐年 vintage，面试中应明确这是静态快照。
7. 可能构成 ex-post bias，因为 2024 决策者当时未必知道未来新增容量；应使用目标日期前已投运容量。
8. 有。只保留当前仍可识别资产会漏掉退役项目，应构建 point-in-time asset universe。
9. 它继承自旧项目的容量优先筛选；当前管线只用其国家、坐标和容量，仍需重新审计选择规则。
10. 无效节点通过元数据和地理检查发现并移除；应把规则固化为 schema 和地理边界测试。
11. 很可能。缺失元数据资产被排除可能偏向大型成熟项目，应报告覆盖容量而不只是节点数。
12. 当前没有完整外部交叉验证；应与官方 registry、ENTSO-E 或业主数据核验位置、容量与投运日期。
13. 将这些属性加入站点级出力模型，并按历史生效日期处理，能显著改善风速到发电的映射。
14. 可用 bidding-zone/资产节点，边由天气相关、物理距离、输电连接或历史 flow 构造，并与非图基线公平比较。
15. 当前 tabular baseline 已有明显增量，空间输入仍很粗；先修数据比增加模型复杂度更可能提升真实 OOS。
16. 只有当站点级/zone 级数据完善、图边有明确含义，且图模型跨 fold 稳定战胜强基线时才重新引入。

## 八、特征工程

1. Lag 1 捕捉短期持续性，lag 2 提供局部变化，lag 7 捕捉周周期；它们是少量、可解释的起点。
2. 7 日代表短期周周期，30 日代表较慢水平；窗口仍需通过训练期内敏感性分析验证。
3. 它保留早期样本但使不同日期的统计精度不一致；应加入有效样本数或在 warm-up 后统一起点。
4. 早期产生 NaN，随后由 train-only 中位数插补，可能引入与样本起点相关的模式。
5. 不足。可比较 14/30 日波动、robust MAD 和尖峰频率，但要控制规格搜索。
6. 当前项目优先验证天气数据链；这些变量是明显遗漏项，也是下一阶段最有价值的数据改进。
7. 正余弦保持周期连续性，避免星期日到星期一或年末到年初出现人为距离跳跃。
8. 影响很小但不严格；可按实际年长计算，或使用 Fourier terms 基于连续时间。
9. 需要，尤其是节假日和工作日对负荷明显；但应使用 point-in-time 市场日历。
10. 它允许每国有不同基准价格水平，Ridge 其余斜率仍默认跨国共享。
11. Dummy 本身经标准化后仍表达类别差异，但系数解释会变化；可在 pipeline 中对数值列和 dummy 分开处理。
12. 未必。统一缩放强调 pooled 分布，分国尺度差异较大时应比较 country-wise normalization 或交互项。
13. 有，天气统计、lags 和滚动均值高度相关；Ridge 正是为稳定共线特征的线性估计而选。
14. Alpha 增大时收缩相关系数，降低方差但增加偏差；由 validation month 选择折中。
15. 尚未形成正式系数稳定报告；应按 fold 输出标准化系数、符号和置信区间。
16. 当前特征数有限，先用预定义 ablation；扩展后应做跨 fold stability selection，而非全样本筛选。
17. 在训练 fold 内预注册候选 family，做嵌套验证，并对最终统计结论做多重检验或 reality-check 修正。
18. 应该。这会把天气变量转成更接近 supply shock 的容量因子或潜在 MWh。
19. 用站点天气经过 turbine/solar 模型得到容量因子，乘历史可用容量，再减负荷预测形成 residual load。
20. 优先加入负荷预报、实际/预报风光出力、燃料与碳价、机组 outages、水文与跨境 flow。

## 九、泄漏与时间戳纪律

1. 时间线是：D-3 预报、D-2 更新、D-1 09:00 决策、随后拍卖 gate close 与出清、D 日交付；所有特征必须在决策前发布。
2. `build_panel` 中断言 `forecast_available_time < decision_time`，但它验证的是代理时间，不是原始 issue timestamp。
3. 同一函数断言 `decision_time < target_date`，防止决策落到交付日之后。
4. 当前只验证自定义代理时间；生产级证据必须来自每个 forecast vintage 的 issue time。
5. 不算。目标日期的日历在决策时已知，是确定性外生信息。
6. 不算，只要使用的是决策前发布、针对目标日的 forecast，而不是目标日 realization。
7. 通常不算数值泄漏，但如果 universe 是事后选择，就存在 sample-selection look-ahead。
8. 可能。应在研究开始前冻结可交易/可预测市场清单，或使用逐日可用 universe。
9. `TrainOnlyTransformer.fit` 只在 fold 的 train 上计算 1%/99% 分位数。
10. 同一 transformer 只用 train 计算中位数；最终 refit 时用 train+validation，不使用 test。
11. 不重新拟合，只使用 train transformer 评估候选 alpha。
12. 只使用 validation month 的 MAE 选择，test 不参与。
13. 超参数已用 validation 选定，train+validation 在真实时间上都早于 test，因此可以重拟合以增加信息。
14. 直接计算不影响单次 fold，但研究过程反复查看 test 后改规格会产生隐性 test overfitting。
15. 是。经过多轮修改后 2024 已不再是严格 untouched holdout，需要新的未来数据做冻结验证。
16. 尝试次数很多且未完整量化，这是 research overfitting 风险；应建立 experiment registry 并预注册最终规格。
17. 旧路线仍在 Git history 和 research history 中，但应保留实验清单和结果索引，单靠 Git 不够方便审计。
18. 理论上代码可恢复，但外部 API vintage 和未版本化环境可能不可重建，因此需要数据 checksum 与 lockfile。
19. 因为错误解析后的时间仍可能满足形式上的大小关系；必须用已知日期样本验证当地时区和市场日。
20. 为每条记录保存 `event_time`、`issue_time`、`ingest_time` 和版本，只允许查询决策时点之前的最新 vintage。

## 十、Walk-forward 与 OOS 设计

1. Expanding window 模拟数据随时间累积并提高有限样本利用率；rolling window 更能适应漂移，应作为稳健性比较。
2. 三个测试期依次是 7-8 月、9-10 月、11-12 月；每个之前用一个月验证，其余更早数据训练。
3. 因为 archive 很短，只能在调参样本量和测试覆盖间折中；一个月仍偏短。
4. 两个月提供连续 OOS，同时形成三个不重叠区间；这不是理论最优，只是数据约束下的设计。
5. 不重叠，代码还检查 `model/target_date/country` 的重复 OOS prediction。
6. 允许，这是标准 walk-forward：在后续真实时点，过去测试期已经成为可用历史。
7. 因为只拼接每个时点当时未参与拟合的测试预测，并确保测试日期不重复。
8. 对最初单次设计是 OOS，但经过反复研究迭代已变成 adaptive OOS；应称 historical walk-forward，并新增 frozen holdout。
9. 能覆盖夏秋冬部分季节和若干价格状态，但不能覆盖多年燃料、危机、水文和政策周期。
10. 历史 forecast revision archive 不覆盖 2022，因此当前实验无法检验能源危机 regime。
11. 按负价、价格分位、波动、季节、燃料和供需状态分层报告误差与改善，并保持足够样本。
12. 先冻结数据处理、特征 family、模型、超参数搜索和主要指标，再一次性解封 2025-2026 holdout。
13. 是，每个 fold 用自己的 validation month 选择 alpha/树参数。
14. 应同时报告 pooled、fold 等权和 country 等权；不同权重回答不同问题。
15. 可用时间阻塞的 outer split，再做 leave-country-out 或层级评估；不能随机打散 country-day。
16. 它检验模型是否学到可迁移机制，而不是只记住国家截距和历史模式。
17. 一日目标的非重叠标签不强制 purge，但滚动特征、发布延迟和多日目标扩展时应设置 embargo。
18. 会增加参数不稳定和 warm-up 缺失；这也是偏好 Ridge 与有限特征的原因。
19. 第一 fold 训练有 616 个 country-day，统计上可拟合小型 Ridge，但对复杂模型仍偏少。
20. 比较 fold 间特征分布、缺失率、残差和系数，并用 PSI、KS、rolling error 与校准监控。

## 十一、Baseline 与 Ablation

1. Persistence 直接用目标国上一交付日的日均价预测下一日，即 `prediction = price_lag1`。
2. 电价水平有强自相关，任何模型若不能战胜 persistence，就没有证明天气或复杂模型提供增量。
3. 7 日均值对快速 regime 变化反应慢，会平滑近期有效价格水平，因此在这段样本中落后。
4. 应加入，例如上一周同一星期的价格；它能更公平检验周季节性是否已解释模型改善。
5. 应加入 country × weekday、滚动季节均值等低复杂度基线，避免把日历信息误归因于天气。
6. 受限于首版范围；AR/SARIMAX、GAM 和 ElasticNet 都是合理强基线，应在冻结 holdout 前补齐。
7. Ridge 对小样本、共线天气统计和可解释性较合适，但是否“足够强”只能通过更多传统模型比较证明。
8. 应加入，但要限定预注册参数预算；否则模型搜索自由度会吞噬短 validation 的可信度。
9. 是。所有模型在每个 fold 使用同一 test frame，合并后还检查 model/date/country 键不重复。
10. 严格说它是 price + calendar + country baseline，名称简化过度；报告应明确 feature list。
11. 其 1.14% 可能来自天气、日历或 country intercept，必须再加 calendar+country-only baseline 才能归因。
12. Revision 是相对变化，缺少价格水平和 regime 条件时很难直接映射绝对电价，因此单独失败并不意外。
13. 对。两者差异就是 day-2 forecast levels，因此 2.82 EUR/MWh 的改善主要来自水平信息。
14. 分别建立 wind-only、solar-only、temperature-only，并做逐 family 删除，保持完全相同 folds 和 preprocessing。
15. 应做。Leave-one-family-out 更适合衡量在其他相关特征存在时的边际贡献。
16. 应增加，尤其用于判断 forecast-level-only 的小改善究竟来自天气还是静态控制。
17. 不足以声称全面优化；这是控制搜索空间的首版，下一步应在 nested validation 中扩大但预注册网格。
18. 限制候选数、跨多个 fold 看参数稳定性，并把最终规格放到全新 frozen holdout。
19. 我会选简单模型，因为目前没有足够证据支付复杂度、维护和不稳定性的成本。
20. 先证明数据和简单机制有增量，再谈结构复杂度；复杂模型必须稳定、显著地战胜强基线。

## 十二、损失函数与指标

1. MAE 与业务单位一致、对尖峰比 MSE 稳健，适合做首要平均误差指标。
2. MAE 衡量典型绝对误差，RMSE 更惩罚尾部，R2 衡量相对常数模型解释度；三者不能互相替代。
3. 若业务重视稳定典型误差可选 HistGradient；若重视尖峰、解释和稳健性，我倾向 Ridge，并继续看 tail metrics。
4. 会。国家均值差异可提高 pooled R2，应补 within-country demeaned R2 与 country-level R2。
5. 应该，它能更直接衡量模型对每国时间变化的解释，而不是横截面价格水平。
6. 负 bias 表示整体低估；在电价尖峰风险管理中可能导致不足报价或风险预算偏低。
7. 会错过最昂贵状态，平均 MAE 好看也可能业务失败，因此必须报告高分位价格日的条件误差。
8. 负价或接近零使百分比误差爆炸且符号难解释，所以不适合作为主要指标。
9. 应报告。电价风险天然需要分布预测，pinball loss、coverage 和 interval width 比点预测更完整。
10. 可使用加权 MAE、Huber、quantile loss 或对高价事件单独建分类/回归模型。
11. 应改为与报价或 PnL 对齐的 utility/loss，例如不对称偏差成本、imbalance cost 或 bid-curve objective。
12. 表中百分比来自所有 pooled OOS country-day 的总 MAE差；bootstrap 则先按日聚合 loss improvement。
13. 不同价格水平下经济意义不同，应补 normalized error、分位 regime 和按市场价值加权的指标。
14. 应报告，尤其是 median、P90/P95 absolute error 和 worst-day error，展示尾部而非只看均值。
15. 画残差 ACF、按预测值分箱的方差、QQ/尾部分布，并做 Ljung-Box 和 robust diagnostics。
16. 最终指标集合应预注册，避免在多个指标中挑一个最漂亮的结论。

## 十三、统计显著性

1. 抽样单位是按日期聚合后的每日平均 loss improvement，block 由连续日期组成。
2. 这样保留同日七国共同冲击，避免把高度相关的国家观测误当成七个独立样本。
3. IID country-day bootstrap 会破坏时间自相关和同日横截面相关，通常低估标准误。
4. 电价误差和天气 regime 连续多日相关，moving block 保留局部时间依赖。
5. 7 天对应自然周和观察到的短期依赖，是可解释起点；不是唯一正确值。
6. 当前没有数据驱动选择，应根据 ACF、自动 block-length 方法和敏感性分析补强。
7. 尚未系统汇报；正式结论应在 3、7、14、28 天下方向一致。
8. 勉强可做短块推断，但尾部精度有限；区间应视为初步而非终局证据。
9. 每次随机选择合法起点，拼接固定长度连续索引，最后截断到原序列长度；不环绕末端。
10. Stationary bootstrap 可减少固定块边界效应，circular block 可更均匀使用末端，应作为稳健性比较。
11. 当前是 percentile interval，不是 BCa；短样本和偏态下 BCa 可能更合适。
12. 它是 bootstrap 样本中平均改善小于等于零的比例。
13. 是单侧，因为问题是 challenger 是否改善；若检验任意差异应使用双侧定义。
14. 估计 0.007 只有约 7 个反例，Monte Carlo 误差不小；最终应增加到至少 10,000 次。
15. 输入是按日期平均后的 loss improvement 时间序列，不是 1,281 个原始误差。
16. 为与一周依赖假设一致设 7，但应由残差相关和敏感性分析支持。
17. 当前实现是基础 Bartlett 权重 HAC，没有小样本或 prewhitening 修正。
18. 它支持“在当前规格下有统计增量”的初步说法，不足以证明 alpha、经济价值或未来稳定性。
19. 需要。尝试多个模型、feature family 和参数后，应做 FDR、SPA/Reality Check 或新 holdout。
20. 存在，尤其项目经历多轮迭代；应保存完整试验账本并使用冻结数据做最后验证。
21. 可能。应加入 calendar-only、seasonal naive、逐月/季节控制和新年份验证，排除天气只是季节代理。
22. 应表述为“小但统计为正的条件增量”，不能用“强信号”或“显著盈利”。
23. 当前先按日期平均处理同日相关；更正式可用 date 与 country two-way cluster 或层级 bootstrap。
24. 时间推断的核心自由度更接近 183，因为七国同日不独立；1,281 只是误差计算样本量。
25. 最可信的统计结论仍需在未参与研究选择的新时期复现，而不只是换标准误方法。

## 十四、结果解释与稳定性

1. 它同时利用强价格持续性、季节控制和目标日天气水平，能解释 persistence 无法捕捉的供需变化。
2. 当前报告未输出系数，因此不能凭印象回答；应生成按 fold 的标准化 coefficient table。
3. 尚未验证。经济符号检查是下一步模型审计，若跨 fold 频繁翻转应降低结论强度。
4. 可能，必须做 daily contribution、top-loss-day 和 leave-extreme-days-out 分析。
5. 尚未报告；这是很有价值的稳健性检查，但删除规则必须事前定义，避免人为清洗。
6. 尚需测试。若结论只在 1%/99% 下成立，说明模型对预处理规格敏感。
7. Metadata 已写入逐行预测但清理后未汇总；应在 fold report 中明确列出每个 alpha。
8. 尚未形成参数稳定表；参数频繁变化会削弱复杂模型可信度。
9. Full Ridge 在三个 fold 都胜过 persistence，但幅度并非单调，最后一段仍有正改善。
10. Revision 增量弱且 regime-dependent，价格或天气关系变化可能使其在最后 fold 不足以抵消噪声。
11. 目前只有 pooled pairwise bootstrap；应逐国报告 revision 增量及区间。
12. 尚未分月报告；短样本下月度结果会很噪，但能发现单月驱动。
13. 尚未通过 family ablation 确定，不能直接说风 revision 最有效。
14. pooled 结果仍是描述事实，但外推性很弱；必须披露集中度并给宏观等权和 leave-one-out。
15. Ridge 理论上会因样本增加更稳定，但要用系数波动、预测方差和 fold 误差实证验证。
16. 树模型对默认算法和数值细节更敏感，Ridge 有闭式凸优化结构，所以版本变化影响更小。
17. 固定代码、数据 checksum、随机种子和依赖 lockfile，并在报告中记录 runtime versions。
18. 随机种子影响树的部分过程和 bootstrap 抽样；Ridge 本身在固定数据下基本确定。
19. 应保存，并同时记录 feature order、schema、训练截止时间和数据 hash，才能真正部署与复现。
20. 目前可通过开源代码和数据重跑，但需要独立人员、不同实现和官方数据源才能称独立复核。

## 十五、Norway 失败与异质性

1. Norway 的水电主导、分区拥塞和国家汇总都与大陆风光市场不同，统一特征很可能错设机制。
2. 水电具有库存和调度弹性，短期风速冲击可被水电和跨区交易吸收，因此本地风信号较弱。
3. 很可能。NO1-NO5 的水文和拥塞差异被平均后，目标与天气匹配会失真。
4. 是。水库水位、inflow、积雪和水电出力应是 Norway 模型的核心状态变量。
5. 互联线路决定水电与大陆价格的耦合程度，约束变化会导致价格 regime 切换。
6. 应比较均值、负价率、波动、尖峰、水文季节和与大陆价格相关性；当前报告还不完整。
7. 部分北欧 forecast 字段缺失略多，可能贡献误差，但需要 country-level missingness 与误差关联验证。
8. 可以计算但不应作为主结果，因为事后排除会夸大表现；只能作为异质性诊断。
9. 是，若没有事前市场结构理由和独立验证，删除 Norway 就是 cherry-picking。
10. 优先拆 bidding zones、加入水文变量和 country interactions；样本足够后再比较独立模型。
11. 能检验模型是否依赖 Norway 自身历史和 dummy，以及机制能否从其他国家迁移。
12. 我会加入 NO-zone 价格、水库水位、inflow、水电出力、互联 flow 和 zone-level weather。

## 十六、从预测到交易或业务价值

1. 目前不能直接换算。需要知道决策如何改变 bid、成交量和 imbalance cost，才能把 EUR/MWh 误差映射为现金价值。
2. 潜在场景是日前拍卖报价、日前与日内价差或发电资产调度，但当前尚未选择并验证具体产品。
3. 这取决于各市场 gate closure 和可用合约；必须逐市场建立 calendar，不能用一个泛化答案。
4. 需要把价格分布与资产边际成本、容量约束结合，形成分价格档位的量价 bid curve，而非只提交点预测。
5. 没有。因此当前不能可信估计 slippage、market impact 或可成交规模。
6. 因为这些是策略收益序列指标，而当前只有价格预测，没有头寸、成交、成本和结算规则。
7. 要求 point-in-time 信号、明确持仓规则、成交假设、成本、容量限制、完整 PnL 和独立 OOS。
8. 不保证。误差改善可能发生在不可交易或低价值状态，也可能方向正确但幅度不足以覆盖成本。
9. 例如在拍卖前以预测与可交易 forward/consensus 的差构造仓位，并用冻结阈值和容量约束；所有输入必须当时可得。
10. 对交易研究，最强 benchmark 应是决策时可交易价格或市场共识；persistence 只适合预测研究。
11. 改为预测 day-ahead minus intraday 的可实现 spread，并严格对齐两个市场的观察、成交和结算时点。
12. 还包括 bid-ask、市场冲击、非成交、imbalance、保证金、容量、税费和 operational constraints。
13. 无法低成本库存使供需冲击即时反映到价格，尾部更厚，仓位和场景风险管理更重要。
14. 可能远超平均日收益，因此要做价格 cap/floor、stress scenarios、CVaR 和单市场风险限额。
15. 用预测分布和经济价值确定仓位，设置国家/产品/总风险限额；stop-loss 不能替代结构性风险控制。
16. 通常需要，否则高波动国家主导风险；也可按 forecast uncertainty 或 expected utility 调整。
17. 模型和回测必须保留负价，并按市场价格限制与结算规则计算非线性损益。
18. 至少需要可交易报价、成交量、bid-ask、拍卖规则、资产约束、imbalance price 和真实发布时点。
19. 至少覆盖多个季节和关键 regime；我倾向 6 至 12 个月，并要求数据与执行稳定。
20. 新 frozen holdout 中无法战胜强市场基准、优势对成本或规格不稳健，都会让我停止或重构路线。

## 十七、旧 ST-GCN 与股票 Alpha 路线

1. 初衷是用图表达风电场之间的空间天气传播，再用时间卷积捕捉天气到市场的滞后。
2. 节点是风电场，边来自距离/聚类，时间维是小时特征，读出层预测金融或电价目标。
3. 只有部分物理含义；距离不等于输电或价格传导，后来发现图结构与目标的经济连接不足。
4. 同一国家价格在多个节点重复，会让模型把复制次数误当空间证据，增加维度却没有新增信息。
5. 四只股票太少，PC1 对个股选择敏感，且天气到股票回报的传导链太长，难以形成稳健横截面结论。
6. 同一个日频结果被多个 6 小时时点重复使用，会制造伪样本、相关残差和过度自信。
7. 没有稳定战胜最强简单基线，这正是最终不保留 ST-GCN 为主模型的原因。
8. 可以把它作为负面研究经历：我识别了伪空间结构和目标错配，并据证据降复杂度，而不是硬讲成功故事。
9. RankIC 弱、成本后优势不足、数据与 universe 风险高，而且天气对短期残余股票回报的机制过于间接。
10. ERA5 包含事后观测同化，不能代表下单时可见的天气预期，因此可能产生信息泄漏。
11. 最重要原则是先把 target、timestamp、baseline 和 point-in-time 数据做对，再考虑模型复杂度。
12. 如果不披露会有这个观感，所以我保留 `research_history.md` 和 Git history，明确写出负面结果。
13. 它记录路线变化、失败原因和保留部分，使面试官能看到研究判断而非只看最终最好数字。
14. 需要站点/zone 级 point-in-time 数据、经济合理图边，以及图模型在多 fold 与新 holdout 中稳定战胜 tabular baseline。
15. 我会优先预测 zone-level wind generation 或跨区价差，再把它作为电价模型输入；它们与图结构更直接。

## 十八、模型选择与复杂度控制

1. Ridge 的天气增量强、三个 fold 都胜过 persistence、可解释且版本稳定；HistGradient 的额外 MAE 改善尚不显著。
2. 观察改善约 0.55 EUR/MWh，但 95% block-bootstrap 区间包含零，所以不能排除抽样噪声。
3. 阈值效应、价格尖峰、天气与国家交互、风功率曲线和拥塞 regime 都可能违背线性。
4. 应。GAM/spline 能表达可解释非线性，是深树或神经网络前更合适的中间基线。
5. 用参数/搜索自由度、训练与维护成本、版本敏感性、解释难度和 OOS 不稳定性共同衡量。
6. 我要求多 fold 一致、bootstrap 区间为正、新 holdout 复现，并且改善足以覆盖业务实施成本。
7. 当前尚未系统报告；Ridge 和小型 HistGradient 都很快，性能瓶颈主要在数据而非推理。
8. 这是已验证且含全部依赖的本机环境名，不代表当前模型使用 GPU；环境名与算法职责应分开解释。
9. 没有。scikit-learn 的 Ridge 和当前 HistGradient 在 CPU 上运行，GPU 只为环境中其他项目可用。
10. 不意味着。最终项目没有 PyTorch 模型，报告明确记录 scikit-learn 版本。
11. 先冻结 feature/fold/metric，只给 LightGBM 一个有限预注册搜索空间，并在 nested validation 后用新 holdout 检验。
12. 不太适合。validation 只有一个月，复杂优化更容易拟合噪声，有限有依据的网格更稳妥。
13. 可能降低方差，但也增加自由度；只有在 out-of-fold stacking 和新 holdout 中稳定改善才值得。
14. 外层模拟真实 OOS，内层只在过去数据做 expanding folds 选择模型，外层测试完全不参与。
15. 它方便审计天气影响、解释异常报价、满足风险治理，也更容易发现数据泄漏和 regime 失效。

## 十九、代码与实现细节

1. `scripts/run_electricity_forecast.py` 加载 `evaluate.main`，后者构建 panel、逐 fold 拟合、生成预测、统计指标和报告。
2. 它把负容量裁为零；若总权重不正，则退回等权坐标，避免除零。
3. 先按节点 metadata 的国家分组，再取节点名与价格 CSV 列名的交集。
4. 该节点被忽略；若整个国家没有匹配列则不会生成该国价格，因此应增加完整映射断言。
5. 它对重复国家列的异常值较稳健，但理想情况应先验证复制列一致，而不是依赖中位数掩盖问题。
6. Pandas 对 value columns 按 mean/std/min/max 聚合，随后把 MultiIndex 拼成 `<原列>_<统计量>`。
7. 会。当前依赖字符串模式识别，应改成显式 schema 和 expected feature assertions。
8. 保存 columns、median、上下分位数和 scaler；`transform` 只能使用 fit 后状态。
9. 中位数仍为空后被填 0，再缩放；这会静默隐藏坏列，应改为直接报错或删除并记录。
10. 按每个特征列分别 winsorize，不是把所有值放在一个全局分布中。
11. 对树不必要，单调线性缩放通常不改变分裂顺序；为统一 pipeline 使用，但可移除以简化。
12. 在 train 拟合 transformer 和每个 alpha，用 validation MAE 选择；再在 train+validation 重拟合最终模型。
13. 代码使用严格 `<`，所以相同 MAE 保留候选列表中最先出现、即较小的 alpha。
14. 当前固定 150 轮且使用默认 early-stopping 行为，版本差异可能改变结果；应显式锁定相关参数。
15. 它假设相同 predictions table，但未逐模型显式 assert key 集完全一致；应新增一致性断言。
16. sklearn metrics 通常报错，因此 pipeline 会失败而不是静默忽略；应提前给出更清晰的 finite-value assertion。
17. 这三列唯一标识某模型在某国家某目标日的 OOS 预测，足以发现 fold 测试重叠。
18. 它防止目标错位，但把浮点值放进 join key 并不理想；更稳妥是只按 date/country 合并后 assert target 相等。
19. 有，序列化或计算差异可能导致匹配失败；应从 merge key 移除浮点目标。
20. 随机抽多个合法起点，每个取连续 `block_size` 个索引，拼接后截断为长度 n。
21. 当前各模型依次消耗同一 RNG，抽样不完全配对；更严格的比较应预生成共同 bootstrap indices。
22. 固定数据、版本和 seed 时可复现；树默认行为和依赖版本仍需锁定。
23. 汇总 JSON 有 runtime 和 feature list，但 CSV 尚无独立 schema/version 文件。
24. 会留下部分 fold 文件；应先写临时目录，全部成功后原子替换最终输出。
25. 有 timeout、三次指数退避、请求间 pause 和 metadata，但没有内容 checksum 与断点续传。
26. 定义 expected fields、units、时间覆盖和 non-null 阈值，下载后 fail-fast，而不是只保留任意非空行。

## 二十、测试与质量保证

1. 覆盖 revision 算术与时点、train-only transformer、fold 边界、pairwise bootstrap 和真实 panel 时间清洁。
2. 测试只用常数路径验证 mean，未覆盖其他统计与配对逻辑；应补逐小时变化样本。
3. 它证明 transform 新值不会改变保存的 median，但未检查 scaler/quantile 精确来自 train，应补带极端 validation 的参数断言。
4. 只检查单个 fold 内 train/val/test 边界，不检查不同 fold 的 test overlap；生产测试应补全。
5. 为允许无大数据的轻量环境运行，但这可能掩盖集成数据缺失；CI 应把 unit 与 required integration job 分开。
6. 会，所以正式 integration test 在数据缺失时应 skip with explicit reason 或 fail，而不是静默 return。
7. 应。它读取真实 32 MB 数据，属于 integration test；unit tests 应使用小 fixture。
8. 缺少当地 market day、23/25 小时、负价、全空列、连续缺失、单位错误和重复时间戳测试。
9. 计算同国列间最大差和不一致比例，并在超阈值时 fail 或生成审计报告。
10. 用日期可识别的 toy price 序列，逐行断言 D 的 lag1 等于 D-1 且决策时间可得。
11. Mock API payload，断言字段、UTC 时间、单位、覆盖范围、lead 列和缺失阈值。
12. 构造 `forecast_available_time >= decision_time` 的样本，确认 `build_panel` 抛出 AssertionError。
13. 应锁定小型 deterministic fixture 的关键表，并用容差比较，不应把随机大数据输出当唯一 golden file。
14. Ridge 数字可设紧容差；树模型应锁版本或用较宽行为阈值，重点断言方向和范围。
15. 是，`raincalib-gpu` 下测试结果为 5 passed。
16. 提交小 fixture，在 CI 创建锁定环境，仅运行 unit/smoke；完整数据集成测试由受控 runner 执行。
17. 从真实 schema 抽取几周、少数国家并去敏，保留边界、缺失和 DST 代表样本。
18. 质量门槛应包括测试、数据 audit、无泄漏断言、输出完整性和可复现 hash，而不只是代码能运行。

## 二十一、复现与工程化

1. `raincalib-gpu` 是当前机器已验证运行环境，`environment.yml` 是跨机器依赖规范；两者职责不同。
2. 不能完全精确，它固定主要包版本范围但未锁全部间接依赖；严格复现还需 lockfile。
3. Patch 版本通常兼容但本项目已看到树模型对版本敏感，因此最终报告应使用完整版本锁定。
4. 应保存 Windows 平台的 `conda-lock` 或 explicit spec，同时保留通用 environment 文件。
5. 记录 Conda 环境、Python、NumPy、pandas 和 scikit-learn 版本。
6. 树的默认行为、数值实现或 early stopping 细节可能变化；这也是运行时版本必须入报告的原因。
7. 目前没有完整 checksum 体系，应为每个 raw 文件记录 SHA-256、大小、覆盖区间和来源 URL。
8. 是，forecast metadata 按国家记录请求 URL、坐标、覆盖和非空小时数。
9. 下载到临时文件，验证 schema、行数、日期、checksum 后原子重命名；失败不覆盖上一版本。
10. `.gitignore` 排除模型/张量和逐行可再生输出，并已检查工作区没有超过 100 MB 的文件。
11. 它们体积较大且可由固定数据和代码重建；小型汇总便于代码审查和回归比较。
12. 代码和部分历史结果在 Git 中，但要精确复现仍需旧数据 hash 与环境；这是当前历史复现边界。
13. 数据量和协作增长后需要。DVC/对象存储可把数据版本与 Git commit 绑定，避免把大文件直接入库。
14. 用 commit SHA、data hash、config hash、environment lock hash 和 seed 生成唯一 ID。
15. 每次运行写独立 experiment ID 目录，完成后更新只读 registry，不覆盖共享路径。
16. 每阶段有显式输入输出和 schema，支持缓存、重跑和独立测试；可用 Make、Snakemake 或轻量 orchestrator。
17. 使用结构化 JSON 日志、单一 config、状态清单和原子输出；失败后从最后通过的数据阶段恢复。
18. 监控数据延迟、缺失、分布漂移、预测范围、残差、coverage、国家级性能和 fallback 使用率。
19. 回退到最近验证的 persistence/seasonal baseline，标记预测降级并告警，不能用陈旧天气静默运行。
20. 对输入和残差做滚动 PSI/KS、missingness 和 conditional error 监控，并按预设阈值触发审查或重训。

## 二十二、简历真实性与表达

1. 不能。当前没有持仓、成交、成本和 PnL，只能写“价格预测研究”。
2. 可以说“预测误差改善在当前 block-bootstrap 下为正”，但应同时给样本期；不能把统计显著等同经济显著。
3. 分母是 persistence 在相同 1,281 个 OOS country-day 上的 pooled MAE 22.657。
4. 主写 Full Ridge 的 21.32%，因为稳健透明；可补充 HistGradient MAE 最低但没有显著战胜 Ridge。
5. 183 是近似独立时间维度，1,281 是误差计算的 country-day 数；两者共同说明样本结构。
6. 明确写“183 complete OOS days across seven countries”，不要只写 1,281 observations。
7. 应主动披露为异质性发现，显示没有只挑正结果，也为下一步 zone/hydrology 改进提供依据。
8. 写成“在 price history 条件下额外降低 MAE 0.14 EUR/MWh，区间为正，但经济价值尚未验证”。
9. 它展示了我能用强基线否定复杂模型、定位伪空间结构并重构研究问题，这是研究判断而非失败包装。
10. 我负责问题定义、代码、验证与解释；Open-Meteo 提供数据，scikit-learn 提供算法实现，工具协助开发但结论需我审计。
11. 运行 `conda run -n raincalib-gpu python scripts/run_electricity_forecast.py`，再核对 summary 和 final report。
12. 查看完整 commit、旧实验删除记录、research history、输出时间和数据 metadata；更严格还需 experiment registry。
13. 从 ST-GCN/股票目标转向 point-in-time 电价目标，并在 HistGradient 略优时仍选择证据更稳的 Ridge。
14. 端到端数据下载、timestamp assertions、train-only preprocessing、walk-forward、bootstrap 和可复现报告。
15. Revision 的 t-stat 2.25 最容易被孤立夸大；它只有 0.14 EUR/MWh，且来自短 OOS 和多轮研究。

## 二十三、压力测试式追问

1. 我不会说证据充分；183 天只支持初步结论，所以使用按日 block inference，并要求新年份 frozen holdout。
2. 公开信息不等于没有预测价值，但也不自动形成 alpha；当前只证明相对简单基线的预测增量，未战胜市场共识。
3. 可能部分已被吸收，因此真正检验应以拍卖前可交易/共识价格为 benchmark，而不是只比昨日价。
4. 区别在 point-in-time 市场时间轴和机制驱动 ablation，但没有交易验证前，它仍只是严谨的预测原型。
5. Ridge 结果稳定，变化只出现在树模型；我已经记录完整版本，并把这种敏感性作为不首选复杂模型的证据。
6. 是，质心是明显粗化。我把它定位为 POC，下一步应改为站点级天气转出力后容量加权。
7. pooled 是整体描述，不代表每国成功；我明确披露 Norway，并计划用 bidding-zone 和水文数据处理。
8. 单看经济量级可能可忽略，所以我不把它作为主卖点；主结果是全部天气特征相对 price-only 的约 2.91 EUR/MWh 增量。
9. 多轮选择会削弱名义 p-value，因此我把它视为描述性证据，并要求实验账本、多重检验和新 holdout。
10. 当前历史预报只有 2024 年，训练确实很短；这说明优先任务是获得更长 immutable forecast archive。
11. 因为这是低成本 POC；生产结论前应迁移到带 issue time 和 model version 的官方 archive。
12. 目前不能完全证明，所以我把 vintage backfill 列为关键数据风险，而不是假装已解决。
13. 当前业务价值有限；日均先验证机制，下一步必须升级到小时曲线和具体拍卖/价差产品。
14. 是重要遗漏，所以当前结果不能解释为完整结构模型；这些正是下一轮最优先数据。
15. Price-only Ridge 已经显著强于简单基线，说明持续性很重要；天气是否有用看 Full 对 Price-only 的 2.91 增量。
16. 不能仅凭 1.14% 下结论，因为该 ablation 含 calendar/country；更有说服力的是 Full 对 Price-only 的配对改善。
17. Full 与 Price-only feature list 的差异是天气 level+revision，配对 bootstrap 显示正增量；仍需 calendar-only 与 family ablation。
18. 当前有 pooled fold table，但跨 fold 区间尚未单独报告，这是应该补的统计输出。
19. 我会先检查数据、regime 和规格稳定性；若预注册的新 holdout 失败，就降低或撤回结论。
20. 如果使用官方 point-in-time archive、强市场共识 baseline 和新多年 holdout 后天气增量消失，当前结论应被推翻。

## 二十四、现场计算与白板题

1. 绝对改善是 `22.657-17.826=4.831` EUR/MWh；相对改善是 `4.831/22.657≈21.32%`。
2. 先形成每日 loss difference `d_t`，估计均值；用 Bartlett 权重汇总 0 至 L 阶自协方差得长期方差，再以 `mean(d)/sqrt(LRV/T)` 计算 t-stat。
3. 重复 B 次：随机抽合法 block 起点，连接若干长度 7 的连续日期，截到 T，计算重采样均值；最后取分位数和小于零比例。
4. 同日国家共享天气、燃料和互联冲击，时间上误差也连续相关；IID 会把相关观测当新增信息，使区间过窄。
5. Ridge 最小化 `||y-Xβ||² + α||β||²`；α 增大使系数收缩、方差下降、偏差上升。
6. `MAE=mean|y-ŷ|`，`RMSE=sqrt(mean(y-ŷ)²)`，`R²=1-SSE/SST`。
7. 模型 A 大多数误差为 1、一次误差 20；模型 B 每次误差为 3。A 可能 MAE 更低但 RMSE 因尾部更高。
8. 令预测等于真实值的十倍，相关系数接近 1，但尺度严重错误，MAE 很大。
9. 低于 cut-in 无出力，中间近似立方上升，到 rated 后平台，高于 cut-out 因安全停机降为零。
10. 图上依次放 forecast vintages、D-1 09:00 决策、gate closure、auction result 和 D 日 23/24/25 小时交付。
11. 以当地 delivery hours 为键，对实际 23 或 25 个小时等权或按 MWh 权重聚合，不能强行补成 24。
12. 先算每国 baseline loss 减 model loss，再对当天可用国家等权平均，得到一个 `d_t`。
13. 同时允许相同日期和相同国家内相关，可用 two-way cluster sandwich，减去交叉重复部分；短面板需小样本修正。
14. 设共同总体斜率，加国家随机截距和随机斜率；国家参数从共同分布收缩，平衡共享与异质性。
15. 最大化预测价格分布下的期望收益减风险与成本，约束数量、机组容量、报价单调性和市场规则。

## 二十五、下一阶段研究计划

1. 一周内优先修正当地 market calendar/DST、加入强 seasonal/market baseline、完成逐国与 feature-family 稳定性报告。
2. 评估并下载带 issue timestamp 的官方 NWP archive，保存 immutable vintages、model version、checksum 和覆盖审计。
3. 在每个历史有效风电场位置查询 forecast，先经功率曲线转容量因子，再按当时在线容量汇总。
4. 接入 TSO/ENTSO-E point-in-time generation forecast，严格区分 forecast、actual 和发布时间。
5. 统一成决策时点快照，分别记录 load、fuel、EUA、outage 和 flow 的 event/issue time，再做增量 ablation。
6. 建立 country/zone × delivery-hour panel，用小时季节性、小时天气路径和多输出/独立小时模型预测整条曲线。
7. 用 quantile regression、NGBoost 或 conformal calibration 输出分位数，并报告 coverage、width 与 tail calibration。
8. 预注册当前最终 pipeline，冻结 2025-2026 数据，一次性评估主要指标，不再根据结果改模型。
9. 每日按真实发布计划生成预测并记录，先不下单；监控延迟、漂移、误差和与市场共识的差。
10. 至少要求多年/多 regime 中稳定战胜强预测基线、区间为正、国家表现可解释且系统可靠。
11. 还需基于可交易 benchmark 的净 PnL、成本/容量压力测试、风险指标和足够长 paper trading。
12. 当误差主要来自遗漏的基本面、空间粗化或时间戳不确定时，应停止调参，先改善数据。
13. 当高质量细粒度输入已建立，简单模型残差仍有稳定空间/时序结构，且复杂模型能在新 holdout 胜出时。
14. 优先购买带原始 issue time 的官方天气 ensemble、市场共识/forward quote、负荷/出力预报和 bidding-zone 基本面。
15. 两周内最可信交付是修正当地市场日与 DST、完成逐 feature/country 稳健性、增加 seasonal/GAM baseline，并冻结下一阶段 protocol。

## 使用提醒

1. 参考回答不是背诵稿，应根据面试岗位压缩成 30 秒、90 秒和 3 分钟三个版本。
2. 被追问数据来源、交易规则或尚未完成的实验时，优先承认边界，再给出可执行验证方案。
3. 当前最安全的主张是“天气预报水平改善了历史价格基线下的 OOS 预测”；不要越界成“已发现可交易 alpha”。
