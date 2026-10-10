# 欧洲日前电价预测项目：面试问题全集

本文档从量化研究、电力市场、机器学习、统计推断和工程实现五个角度，对当前项目进行压力测试。问题以当前代码与结果为准，不假设项目已经形成可交易策略。

## 项目事实卡

- 目标：预测下一交付日的国家日均日前电价，单位为 EUR/MWh。
- 国家：比利时、德国、丹麦、法国、爱尔兰、荷兰、挪威。
- 决策时点：交付日前一日 09:00 UTC。
- 天气数据：Open-Meteo Previous Runs 的历史预报。
- 预报水平：同一交付小时的 day-2 forecast。
- 预报修正：同一交付小时的 day-2 forecast 减 day-3 forecast。
- 天气变量：100 米风速、短波辐射、2 米气温。
- OOS：2024 年 7 月至 12 月，共 183 个完整 UTC 日、1,281 个 country-day。
- 主要模型：Persistence、7 日均值、Ridge、HistGradientBoosting。
- 当前结果：Full Ridge MAE 17.83，较 persistence 改善 21.32%；HistGradient Full MAE 17.30。
- Revision 增量：Price + Revision 相对 Price-only 改善约 0.14 EUR/MWh。
- 当前结论：天气预报水平有明显增量，forecast revision 只有较小的条件增量；项目尚未证明可交易盈利。

## 一、30 秒项目介绍

- [ ] 请用 30 秒介绍项目，不能使用“AI”“大数据”“端到端”这些泛化词。
- [ ] 这个项目解决的业务问题是什么？谁会使用预测结果？
- [ ] 为什么从股票 alpha 转向电价预测？
- [ ] 项目最核心、最可验证的研究假设是什么？
- [ ] 项目最重要的结果是什么？请同时说绝对改善和相对改善。
- [ ] 你认为最可信的模型是哪一个？为什么不是 MAE 最低的模型？
- [ ] 这个项目目前最大的局限是什么？
- [ ] 如果简历只能写一行，你会怎么写而不过度宣传？
- [ ] 哪个结论是正结果，哪个结论是负结果？
- [ ] 如果面试官说“这只是一个 sklearn 作业”，你怎么回应？

## 二、研究问题与因果机制

- [ ] 天气预报为什么应该影响日前电价？完整传导链条是什么？
- [ ] 风速、太阳辐射、气温分别通过哪些供需渠道影响电价？
- [ ] 为什么预测天气的 revision 可能比天气水平更有信息？
- [ ] 如果市场参与者也能看到同样的天气预报，为什么这个信号不会立即被价格吸收？
- [ ] 你的模型是在预测基本面，还是在预测市场对基本面的反应误差？
- [ ] 预报修正与日前电价之间应该是线性关系吗？
- [ ] 风速增加为什么不一定总导致电价下降？
- [ ] 极高风速触发风机切出时，现有特征会如何失效？
- [ ] 气温对价格的影响为什么可能随季节、国家和时段改变？
- [ ] 短波辐射如何映射为光伏发电？为什么直接使用辐射可能不够？
- [ ] 负电价出现时，天气与价格的关系会发生什么变化？
- [ ] 燃气、煤炭、碳价和机组边际成本在你的机制中处于什么位置？
- [ ] 水电库存、核电检修和跨境输电约束会怎样混淆天气效应？
- [ ] 你如何区分真正的经济机制与季节性相关性？
- [ ] 该项目是预测研究、因果研究，还是交易研究？为什么？

## 三、目标变量与决策时点

- [ ] “下一交付日国家日均日前电价”精确定义是什么？
- [ ] 原始价格是日前拍卖价、现货指数、节点价还是分区价？
- [ ] 价格对应哪个交易所或数据供应商？
- [ ] 目标日的 24 个小时如何聚合为一个均价？
- [ ] 夏令时切换日只有 23 小时或有 25 小时，你如何计算日均价？
- [ ] 为什么选择日均价格，而不是逐小时价格？
- [ ] 日均目标会掩盖早晚高峰和极端小时吗？
- [ ] 为什么不预测 peakload、baseload 或小时曲线？
- [ ] 09:00 UTC 对每个国家分别对应几点当地时间？
- [ ] 09:00 UTC 是否始终早于各市场日前拍卖 gate closure？
- [ ] CET/CEST 切换是否导致固定 UTC 决策时点的经济含义变化？
- [ ] 目标价格在现实中何时首次可知？
- [ ] 如果价格文件中的时间戳是发布时点而不是交付时点，会发生什么？
- [ ] `price_lag1` 在决策时点是否已经完整可知？请画时间线证明。
- [ ] 周末和公共假期的拍卖安排是否与工作日相同？
- [ ] 为什么目标没有做去均值、标准化或价格变化变换？
- [ ] 预测价格水平会不会让 persistence 看起来过强或 R2 看起来过高？
- [ ] 如果改成预测 persistence residual，结果可能怎样变化？
- [ ] 负价格和尖峰价格如何进入损失函数？
- [ ] 这个目标如何转化为一个可执行的报价或交易决策？

## 四、国家与市场范围

- [ ] 为什么选择这七个国家？
- [ ] 七国是事前确定，还是根据数据完整性事后筛选？
- [ ] 为什么没有西班牙、意大利、瑞典、芬兰或英国？
- [ ] 缺少的国家是否会导致选择偏差？
- [ ] 德国价格区应如何处理 DE-LU 市场定义？
- [ ] 挪威存在多个 bidding zones，为什么可以汇总成国家价格？
- [ ] 丹麦 DK1 和 DK2 的结构差异如何处理？
- [ ] 爱尔兰 I-SEM 与欧洲大陆市场机制不同，是否应该单独建模？
- [ ] 国家 dummy 能否充分吸收市场结构差异？
- [ ] 国家之间价格高度相关，1,281 个 country-day 是否等于 1,281 个独立样本？
- [ ] 如果德国样本对 pooled 指标贡献最大，如何防止结论被单一市场主导？
- [ ] 为什么使用统一模型，而不是每个国家一个模型？
- [ ] 统一模型与分国模型各自的偏差和方差权衡是什么？
- [ ] 是否应该使用层级模型或 country-specific slopes？
- [ ] 跨境互联会导致邻国天气影响本国价格，你的特征如何体现？
- [ ] 如果增加 bidding-zone 粒度，需要怎样重构数据和评估？

## 五、电价数据质量

- [ ] 原始价格文件是谁生成的？可否追溯到官方市场数据？
- [ ] 为什么同一国家价格曾被复制到多个风电节点？
- [ ] 当前代码如何把重复节点列合并为国家价格？
- [ ] 为什么选择中位数而不是均值或第一列？
- [ ] 如果同国节点列并不完全相同，中位数会隐藏什么问题？
- [ ] 是否检查过同国重复列之间的最大差异？
- [ ] 缺失价格如何处理？
- [ ] 连续缺失与随机缺失对结果的影响有什么不同？
- [ ] 是否存在重复时间戳、非单调时间戳或错误时区？
- [ ] 是否检查过价格单位始终是 EUR/MWh？
- [ ] 非欧元国家的价格是否完成货币转换？使用哪个汇率时点？
- [ ] 异常价格是市场真实尖峰还是数据错误？如何区分？
- [ ] 为什么 winsorize 输入特征却不 winsorize 目标？
- [ ] 1% 和 99% winsorization 阈值为什么合理？
- [ ] 如果把极端价格裁剪掉，是否正好删除了最有经济价值的事件？
- [ ] 每个 OOS fold 的价格分布是否发生漂移？
- [ ] 训练集和测试集的负价格比例分别是多少？
- [ ] 价格数据的许可证是否允许在 GitHub 中发布？
- [ ] 当前 32 MB CSV 是否是最合适的存储格式？
- [ ] 如何用 Parquet、schema 和数据校验提升数据层可靠性？

## 六、天气预报数据与 Forecast Revision

- [ ] 为什么从 ERA5 realized weather 改成历史天气预报？
- [ ] 使用 realized weather 会造成哪一种泄漏？
- [ ] Open-Meteo Previous Runs 的底层天气模型是什么？会随时间改变吗？
- [ ] Open-Meteo 是否提供原始 issue timestamp？
- [ ] `previous_day2` 和 `previous_day3` 的精确定义是什么？
- [ ] 为什么 revision 定义为 day-2 减 day-3，而不是 day-1 减 day-2？
- [ ] day-2 的所有交付小时为什么被认为在 D-2 23:00 UTC 前可用？
- [ ] 该可用时间是假设、文档保证还是逐条元数据验证？
- [ ] 如果供应商在历史归档中回填或重算预报，会不会产生 revision vintage 泄漏？
- [ ] 历史预报 API 是否存在模型升级导致的非平稳性？
- [ ] 为什么只有 2024 年 3 月以后有可用的风速和辐射历史？
- [ ] 不同国家的 day-2/day-3 缺失率分别是多少？
- [ ] 缺失值是否集中于特定月份、国家或变量？
- [ ] 训练中位数插补是否会人为制造“无 revision”信号？
- [ ] 是否加入 missingness indicator？为什么？
- [ ] revision 使用绝对差还是相对差更合理？
- [ ] 风速 revision 的单位与经济影响是否应做非线性转换？
- [ ] 为什么不使用集合预报的 spread、分位数或不确定性？
- [ ] 为什么不使用预报误差，即 forecast minus realization？
- [ ] 预报误差在真实决策时点是否可得？
- [ ] 只选三个天气变量的依据是什么？
- [ ] 为什么没有云量、降水、气压、风向和露点？
- [ ] revision 的日内 mean/std/min/max 各自代表什么？
- [ ] 对 day-2 和 day-3 分别聚合后再相减，与逐小时相减后再聚合是否等价？
- [ ] 对标准差而言，这两种 revision 构造为什么不等价？当前代码使用哪一种？
- [ ] 预报 revision 单独表现很差，是否说明研究假设失败？
- [ ] Price + Revision 的 0.14 EUR/MWh 增量是否具有经济意义？
- [ ] revision 结果是否由单一变量、国家或月份驱动？
- [ ] 如果 API 明天停止提供 previous runs，如何替代？

## 七、空间表示与风电场元数据

- [ ] 为什么用风电场容量加权的国家质心查询天气？
- [ ] 一个国家只使用一个经纬度会损失多少空间信息？
- [ ] 容量加权质心可能落在海上或无代表性的地点吗？
- [ ] 为什么不在每个风电场位置查询预报后再容量加权？
- [ ] 风电功率是风速的非线性函数，先平均坐标再取风速是否合理？
- [ ] 当前容量数据对应哪个年份？
- [ ] 使用当前装机容量回看 2024 年是否构成 ex-post 信息？
- [ ] 退役、新增和容量扩建如何处理？
- [ ] 风电场清单是否存在 survivor bias？
- [ ] `processed_nodes.csv` 的 100 个节点如何筛选？
- [ ] “Outside the Northeast”等无效节点是如何发现并处理的？
- [ ] 元数据缺失的节点是否系统性地被排除？
- [ ] 风电场经纬度和容量是否经过外部交叉验证？
- [ ] 如何加入 offshore/onshore、hub height 和 turbine type？
- [ ] 如果恢复空间模型，你会使用什么图结构？
- [ ] 现阶段为什么不应该重新引入 ST-GCN？

## 八、特征工程

- [ ] 价格特征为什么选择 lag 1、2、7？
- [ ] 为什么滚动均值使用 7 日和 30 日？
- [ ] `price_roll7` 的 `min_periods=4` 有什么影响？
- [ ] `price_roll30` 的 `min_periods=14` 会让早期样本产生什么缺失模式？
- [ ] 价格波动率只用 7 日是否足够？
- [ ] 为什么没有加入小时负荷预测、燃料价格和碳价？
- [ ] 星期和年内季节性为什么用正余弦编码？
- [ ] 一年按 366 天编码会对非闰年产生什么误差？
- [ ] 是否需要月份、节假日和工作日特征？
- [ ] country dummy 在 Ridge 中如何改变截距？
- [ ] country dummy 与统一标准化如何交互？
- [ ] 所有国家共享同一特征缩放是否合理？
- [ ] 输入变量之间是否存在严重共线性？
- [ ] Ridge alpha 如何缓解共线性？
- [ ] 是否检查过 Ridge 系数的符号、稳定性和经济合理性？
- [ ] 为什么没有做 feature stability selection？
- [ ] 如果加入大量天气变量，如何控制 multiple testing？
- [ ] 是否应该按装机容量把天气变量转换为潜在发电量？
- [ ] 如何构造风功率曲线、solar capacity factor 和 residual load？
- [ ] 最值得新增的五类基本面特征是什么？为什么？

## 九、泄漏与时间戳纪律

- [ ] 请在白板上画出 feature time、decision time、auction close 和 delivery time。
- [ ] 代码中哪一行保证天气特征早于决策时点？
- [ ] 代码中哪一行保证决策时点早于交付日？
- [ ] 这些断言验证的是实际发布时间还是你定义的代理时间？
- [ ] `target_date` 被用于构造星期和年内特征，这算泄漏吗？
- [ ] 使用目标日天气预报是否算泄漏？为什么不算？
- [ ] 用完整数据创建 country dummy 是否算训练外信息？
- [ ] 使用全样本确定国家 universe 是否构成 look-ahead bias？
- [ ] winsorization 阈值在哪里拟合？
- [ ] 中位数插补参数在哪里拟合？
- [ ] StandardScaler 在验证集上是否重新拟合？
- [ ] alpha 选择使用了哪些样本？
- [ ] 最终模型为什么可以在 train + validation 上重拟合？
- [ ] 测试集是否影响了特征选择、模型选择或研究方向？
- [ ] 反复查看 2024 OOS 后修改模型，会不会把 OOS 变成隐性训练集？
- [ ] 你做过多少次实验？如何估计 research overfitting？
- [ ] 删除旧实验代码后，如何审计曾经尝试过的规格？
- [ ] Git 历史能否完整重建研究路径？
- [ ] 如果价格文件的时区解析错误，现有断言为什么可能仍然通过？
- [ ] 如何构建 point-in-time 数据库，而不是依赖静态 CSV？

## 十、Walk-forward 与 OOS 设计

- [ ] 为什么选择 expanding-window，而不是 rolling-window？
- [ ] 三个 fold 的训练、验证和测试边界分别是什么？
- [ ] 为什么每个验证期只有一个月？
- [ ] 为什么每个测试期是两个月？
- [ ] 三个测试区间是否有重叠？
- [ ] 后一个 fold 的训练集包含前一个 fold 的测试期，这是否允许？
- [ ] 为什么合并三个测试期后称为 183 天 OOS？
- [ ] 在整个研究迭代过程中，这 183 天还是真正 untouched OOS 吗？
- [ ] 只有半年 OOS 能覆盖哪些市场状态？不能覆盖哪些？
- [ ] 2022 年能源危机为什么没有进入当前 forecast-revision OOS？
- [ ] 如何验证模型在高价、负价和低波动 regime 中的稳定性？
- [ ] 如果使用 2025 至 2026 数据做最终 holdout，你会冻结哪些内容？
- [ ] fold 之间模型超参数是否独立选择？
- [ ] pooled OOS 指标是否应按 fold 等权，而不是按样本数加权？
- [ ] 国家和日期两个维度如何同时进行 cross-validation？
- [ ] leave-one-country-out 测试能回答什么问题？
- [ ] 是否应该加入 embargo 或 purge gap？在本项目中为什么需要或不需要？
- [ ] 训练窗口太短会如何影响 30 日滚动特征和 Ridge 估计？
- [ ] 当前最早训练期有多少个有效 country-day？够不够？
- [ ] 你如何检测 covariate shift 和 concept drift？

## 十一、Baseline 与 Ablation

- [ ] persistence baseline 的定义是什么？
- [ ] 为什么 persistence 是电价水平预测的必要基线？
- [ ] 7 日滚动均值为什么表现比 persistence 差？
- [ ] 是否应该加入 seasonal naive，例如上一周同一天？
- [ ] 是否应该加入按国家、星期和月份的历史均值？
- [ ] 为什么没有 ARIMA、SARIMAX、Lasso、ElasticNet 或 GAM？
- [ ] 线性 Ridge 是否足以作为传统强基线？
- [ ] 为什么没有加入 LightGBM、XGBoost 或 CatBoost？
- [ ] 模型比较是否使用完全相同的测试样本？
- [ ] Price-only ablation 包含 calendar 和 country dummy，名称是否容易误导？
- [ ] Forecast-level-only 也包含 calendar 和 country dummy，如何解释其 1.14% 改善？
- [ ] Revision-only 失败是否可能由遗漏 price history 导致？
- [ ] Full Ridge 相对 Price + Revision 的增量主要来自 forecast level，对吗？
- [ ] 如何分解 wind、solar、temperature 三个 family 的独立贡献？
- [ ] 是否做过 leave-one-feature-family-out ablation？
- [ ] 是否应增加 country-only 和 calendar-only baseline？
- [ ] HistGradient 只尝试两组参数，为什么足够或不够？
- [ ] 你如何防止对 validation month 过拟合？
- [ ] 如果一个复杂模型只改善 0.5 EUR/MWh，但区间跨零，你会选哪个？
- [ ] ST-GCN 没有战胜 baseline 对当前模型选择有什么启示？

## 十二、损失函数与指标

- [ ] 为什么以 MAE 作为主要模型选择指标？
- [ ] MAE、RMSE 和 R2 分别回答什么问题？
- [ ] HistGradient 的 MAE 更好，但 Ridge 的 RMSE 和 R2 更好，如何选择？
- [ ] pooled R2 是否混入了国家间平均价格差异？
- [ ] 是否应该报告每国去均值后的 within-country R2？
- [ ] bias 为负意味着什么？
- [ ] 系统性低估高价事件会产生什么业务后果？
- [ ] Pearson correlation 高是否代表预测误差小？
- [ ] 为什么没有报告 MAPE？负价格和接近零价格会造成什么问题？
- [ ] 是否应报告 pinball loss 或概率预测校准？
- [ ] 如果用户更关心尖峰，应该使用什么损失函数？
- [ ] 如果用户更关心拍卖报价收益，离线损失函数应该如何改变？
- [ ] 21.32% MAE 改善是 pooled 改善还是每日改善的均值？
- [ ] 绝对改善 4.83 EUR/MWh 在不同价格水平下意义相同吗？
- [ ] 是否报告 median absolute error 和误差分位数？
- [ ] 如何检查残差的异方差、自相关和厚尾？

## 十三、统计显著性

- [ ] bootstrap 的基本抽样单位是什么？
- [ ] 为什么先按日期聚合 country-level loss improvement？
- [ ] 为什么不能把 1,281 个 country-day 当成独立样本普通 bootstrap？
- [ ] moving-block bootstrap 为什么比 IID bootstrap 更合适？
- [ ] block length 为什么选 7 天？
- [ ] 是否根据自相关函数或数据驱动方法选择 block length？
- [ ] block length 改为 3、14、28 天时结论是否稳定？
- [ ] 183 天样本对 block bootstrap 是否足够？
- [ ] 当前 moving-block 实现如何处理序列末端？
- [ ] 为什么不用 circular block bootstrap 或 stationary bootstrap？
- [ ] bootstrap 区间是 percentile interval 还是 BCa interval？
- [ ] `p_le_zero=0.009` 的精确定义是什么？
- [ ] 这个 bootstrap p-value 是单侧还是双侧？
- [ ] 1,000 次 bootstrap 对估计 0.007 的精度够吗？
- [ ] Newey-West t-stat 的输入序列是什么？
- [ ] Newey-West lag 为什么也设为 7？
- [ ] Newey-West 方差估计是否对小样本做了修正？
- [ ] t-stat 2.25 是否足以宣称 alpha 或经济显著性？
- [ ] 同时比较多个模型和特征组合，是否需要 multiple-testing correction？
- [ ] 是否存在 data snooping 和 White's Reality Check 问题？
- [ ] Full Ridge 相对 persistence 的显著性强，是否可能只是 forecast level 的季节代理？
- [ ] revision 增量区间很窄但绝对值很小，应该如何表述？
- [ ] country-level 横截面相关如何进入标准误？
- [ ] 是否应使用 two-way cluster standard errors（date 与 country）？
- [ ] 如果只剩 183 个每日观测，自由度应该按 183 还是 1,281 计算？

## 十四、结果解释与稳定性

- [ ] Full Ridge 为什么能稳定战胜 persistence？
- [ ] 哪些 Ridge 系数贡献最大？
- [ ] 系数符号是否符合基本面逻辑？
- [ ] 结果是否由 2024 年某几次尖峰驱动？
- [ ] 删除误差最大的 5 天后结论是否仍成立？
- [ ] winsorization 分位数改变后结果是否稳定？
- [ ] Ridge alpha 在三个 fold 中分别是什么？
- [ ] HistGradient 的最佳参数在三个 fold 中是否一致？
- [ ] 三个 fold 的表现是否单调改善？
- [ ] 为什么 Price + Revision 在最后一个 fold 可能不如 persistence？
- [ ] revision 增量在每个国家分别是多少？
- [ ] revision 增量在每个月分别是多少？
- [ ] 风、太阳和温度 revision 中到底哪一类有效？
- [ ] 如果只有一个国家显著，pooled 结论还能成立吗？
- [ ] 训练样本增加时模型是否变得更稳定？
- [ ] 库版本变化为什么改变 HistGradient 结果，却没有改变 Ridge？
- [ ] 如何保证换机器后结果可复现？
- [ ] 随机种子对 HistGradient 和 bootstrap 分别影响什么？
- [ ] 是否应该保存 fitted model、transformer 和 feature schema？
- [ ] 当前结果能否通过独立实现复核？

## 十五、Norway 失败与异质性

- [ ] 为什么 Full Ridge 在 Norway 不如 persistence？
- [ ] Norway 的水电占比会如何削弱风速信号？
- [ ] 国家均价是否错误合并了 NO1 至 NO5 的分区差异？
- [ ] 水库水位和水文 inflow 是否是 Norway 的关键遗漏变量？
- [ ] Norway 与欧洲大陆的互联容量如何影响价格？
- [ ] Norway 的价格分布和其他六国有何不同？
- [ ] Norway 是否存在更多缺失 forecast 数据？
- [ ] 将 Norway 排除后 pooled 结果改善多少？
- [ ] 事后删除 Norway 是否构成 cherry-picking？
- [ ] 应该为 Norway 建独立模型还是加入交互项？
- [ ] leave-one-country-out 训练能否识别 Norway 的结构差异？
- [ ] 如果项目部署要求覆盖 Norway，你下一步会加什么数据？

## 十六、从预测到交易或业务价值

- [ ] MAE 改善如何转化为经济价值？
- [ ] 你能在哪个市场、以什么产品交易该预测？
- [ ] 09:00 UTC 时有哪些合约或拍卖仍可交易？
- [ ] 如果目标是日前拍卖价，如何基于预测提交 bid curve？
- [ ] 是否拥有真实的成交量、深度、价差和交易成本数据？
- [ ] 为什么当前项目不能报告 Sharpe、turnover 和 max drawdown？
- [ ] 若简历声称“策略盈利”，面试官应要求哪些证据？
- [ ] 电价预测误差下降是否保证交易 PnL 上升？
- [ ] 如何定义一个不依赖未来信息的 trading rule？
- [ ] benchmark 应该是 persistence、市场共识预报还是可交易 forward price？
- [ ] 如果交易 day-ahead 与 intraday spread，目标应该如何改写？
- [ ] 手续费之外还需考虑哪些成本？
- [ ] 电力市场的非存储性如何改变风险管理？
- [ ] 单日极端事件可能造成多大尾部风险？
- [ ] 如何做 position sizing、限仓和 stop-loss？
- [ ] 是否需要按国家波动率归一化头寸？
- [ ] 如何处理价格上下限和负价格？
- [ ] 一个可交易回测至少需要哪些新增数据？
- [ ] 上线前你希望看到多长的 paper-trading 记录？
- [ ] 什么结果会让你决定停止这条研究路线？

## 十七、旧 ST-GCN 与股票 Alpha 路线

- [ ] 最初为什么选择 ST-GCN？
- [ ] 图中的节点、边和时间维分别代表什么？
- [ ] 原图结构是否有明确经济或物理含义？
- [ ] 为什么把国家电价复制到节点会制造伪空间信息？
- [ ] 四只股票的 PC1 目标有什么根本问题？
- [ ] 为什么重复的 6 小时样本配日频标签会夸大样本量？
- [ ] ST-GCN 是否战胜过最强简单基线？
- [ ] 如果没有，为什么仍值得在简历中提到？
- [ ] 71 股票 cross-sectional RankIC 路线为什么被放弃？
- [ ] realized ERA5 对股票预测为什么不满足 point-in-time 要求？
- [ ] 从失败路线中学到的最重要研究原则是什么？
- [ ] 删除旧代码是否让项目看起来在隐藏失败？
- [ ] `research_history.md` 如何确保负面结果仍可审计？
- [ ] 在什么证据出现后，你才会重新考虑图神经网络？
- [ ] 如果恢复图模型，应该预测电价、发电量还是跨区价差？

## 十八、模型选择与复杂度控制

- [ ] 为什么 Full Ridge 是当前首选，而不是 MAE 更低的 HistGradient？
- [ ] HistGradient 相对 Full Ridge 的 MAE 改善区间为什么跨零？
- [ ] Ridge 的线性假设在哪些区域最可能失败？
- [ ] 是否应加入 spline 或 GAM 作为透明的非线性基线？
- [ ] 模型复杂度的成本如何量化？
- [ ] 你如何定义“复杂模型值得”的最低改善门槛？
- [ ] 是否比较过参数量、训练时间和推理延迟？
- [ ] 为什么当前 CPU 模型仍运行在名为 `raincalib-gpu` 的环境中？
- [ ] GPU 在当前 Ridge/HistGradient 实验中是否实际被使用？
- [ ] 环境中 PyTorch 的存在是否意味着项目使用了深度学习？
- [ ] 如果增加 LightGBM，如何避免扩大模型搜索空间导致过拟合？
- [ ] Bayesian optimization 是否适合只有一个月 validation 的场景？
- [ ] ensemble 是否会改善稳定性，还是只增加自由度？
- [ ] 如何用 nested walk-forward 做更严格的模型选择？
- [ ] 简单模型的可解释性在电力交易中有什么实际价值？

## 十九、代码与实现细节

- [ ] 请从入口脚本开始解释整个调用链。
- [ ] `build_country_locations` 如何处理零容量国家？
- [ ] `load_country_hourly_prices` 如何选择国家对应列？
- [ ] 如果节点名与价格列名不匹配会发生什么？
- [ ] 为什么价格聚合使用 median(axis=1)？
- [ ] `daily_forecast_features` 如何生成多层聚合列名？
- [ ] revision 特征是否可能因列名规则意外漏掉？
- [ ] `TrainOnlyTransformer` 的 fit/transform 状态有哪些？
- [ ] 如果训练列全为空，当前代码会如何处理？
- [ ] winsorization 是逐列还是全局进行？
- [ ] 为什么 StandardScaler 对 tree model 也执行？是否必要？
- [ ] `ridge_predict` 如何选择 alpha？
- [ ] alpha 相同 MAE 时如何打破平局？
- [ ] `hist_gradient_predict` 为什么没有 early stopping 的显式控制？
- [ ] `metric_table` 如何保证 baseline 和模型样本完全对齐？
- [ ] 如果某模型有 NaN prediction，指标函数会怎样？
- [ ] duplicate OOS prediction 检查的 key 为什么是 model/date/country？
- [ ] bootstrap merge 为什么包含 `target_price` 作为 key？
- [ ] 浮点目标作为 merge key 是否有风险？
- [ ] `moving_block_indices` 是怎样生成 block 的？
- [ ] 不同模型共享同一个 bootstrap 抽样序列吗？应该共享吗？
- [ ] 当前随机数生成器的使用是否保证可复现？
- [ ] 输出 CSV 是否包含 schema/version 信息？
- [ ] 如果运行中途失败，输出目录会不会留下半成品？
- [ ] API 下载是否有缓存、重试、限速和校验？
- [ ] 如何防止 API 返回 schema 改变后静默产生错误？

## 二十、测试与质量保证

- [ ] 当前五个测试分别覆盖什么风险？
- [ ] revision toy test 为什么只检查 mean，没有检查 std/min/max？
- [ ] train-only transformer test 是否真正证明 scaler 没有泄漏？
- [ ] fold test 是否检查了 fold 之间的 OOS 重叠？
- [ ] real panel test 为什么允许 forecast 文件不存在时直接 return？
- [ ] 这会不会让 CI 在缺数据时仍显示通过？
- [ ] 是否应该把真实数据测试标记为 integration test？
- [ ] 缺少哪些针对 DST、负价格和缺失值的测试？
- [ ] 如何测试同国重复价格列确实一致？
- [ ] 如何测试 target 和 lag 没有 off-by-one？
- [ ] 如何测试 API 返回字段和单位？
- [ ] 如何构造一个明确会泄漏的反例，确保断言能抓住？
- [ ] 是否应加入 deterministic golden-file test？
- [ ] scikit-learn 升级改变 HistGradient 结果后，测试应如何处理？
- [ ] 哪些指标应该设回归容差，哪些不应该锁死？
- [ ] 当前测试是否在 `raincalib-gpu` 中通过？
- [ ] 如何在 GitHub Actions 中运行而不提交大数据？
- [ ] 如何做小型 fixture，使 CI 不依赖 32 MB 价格文件？

## 二十一、复现与工程化

- [ ] 为什么同时保留 `raincalib-gpu` 和 `environment.yml`？
- [ ] 当前 `environment.yml` 是否能精确复现所有间接依赖？
- [ ] 为什么只固定 major/minor，而不是完整 lockfile？
- [ ] 是否应保存 `conda-lock` 文件？
- [ ] 当前 report 记录了哪些运行时版本？
- [ ] 为什么 scikit-learn 版本会改变 HistGradient 的结果？
- [ ] 原始数据是否有 checksum？
- [ ] 下载元数据是否记录每个 API URL？
- [ ] 如何验证下载文件没有被部分覆盖或损坏？
- [ ] 大于 100 MB 文件如何管理？
- [ ] 为什么将逐行预测忽略、但保留汇总指标？
- [ ] 删除旧输出后如何复现历史数字？
- [ ] 是否需要 DVC、LakeFS 或对象存储？
- [ ] 模型、数据和代码版本如何形成唯一 experiment ID？
- [ ] 如果多人协作，如何避免覆盖同一输出目录？
- [ ] 如何把 pipeline 拆成 download、validate、feature、train、evaluate 阶段？
- [ ] 如何加入结构化日志、配置文件和失败恢复？
- [ ] 生产推理需要哪些监控指标？
- [ ] 数据延迟或 API 失败时的 fallback 是什么？
- [ ] 如何检测线上 feature drift 和性能衰减？

## 二十二、简历真实性与表达

- [ ] 简历中能否写“构建了可盈利电力交易策略”？为什么？
- [ ] 能否写“显著预测欧洲电价”？“显著”指统计还是经济显著？
- [ ] 21.3% 改善的分母是什么？
- [ ] 你会在简历中写 HistGradient 还是 Ridge 的结果？
- [ ] 为什么应该同时报告 183 天和 1,281 country-day？
- [ ] 如何避免让人误以为有 1,281 个独立日？
- [ ] 是否应该主动写出 Norway 失败？
- [ ] 如何描述 revision 的“小而显著”而不夸大？
- [ ] 如何解释旧 ST-GCN 没成功却仍有研究价值？
- [ ] 哪些工作是你亲自完成的，哪些由工具或 API 完成？
- [ ] 如果被要求现场复现一个结果，你会运行什么命令？
- [ ] 面试官如何验证你的 Git 历史没有事后改写故事？
- [ ] 项目中最能体现研究判断、而不是编码能力的决定是什么？
- [ ] 项目中最能体现工程能力的部分是什么？
- [ ] 哪个数字你最不愿意被面试官单独摘出来？为什么？

## 二十三、压力测试式追问

- [ ] 你只有 183 天 OOS，凭什么谈统计显著？
- [ ] 你用了公开天气预报，市场早就知道，alpha 在哪里？
- [ ] 你预测的是已经被日前市场消化的信息，不是吗？
- [ ] 你没有交易回测，这个项目和 Kaggle 回归有什么区别？
- [ ] 为什么最佳模型随着 sklearn 版本变化？这还算可复现吗？
- [ ] 你的天气只是一个国家质心，物理意义是否过于粗糙？
- [ ] 你把 Norway 做差了，为什么还要 pooled 汇报？
- [ ] revision 只改善 0.14 EUR/MWh，这是否可以忽略？
- [ ] 你做了这么多版本后，bootstrap p-value 还有意义吗？
- [ ] 如果把 2024 留作真正未见测试集，你还有多少训练数据？
- [ ] 你为什么没有使用官方 NWP issue-time 数据？
- [ ] 你如何证明 Open-Meteo 历史预报没有回填？
- [ ] 你的目标是国家日均价，但电力市场按小时结算，业务价值在哪里？
- [ ] 你没有 load forecast、fuel、carbon 和 outages，模型是不是严重遗漏变量？
- [ ] Ridge 表现好是否只是因为价格高度自相关？
- [ ] Forecast-level-only 只比 persistence 好 1.14%，天气真的有用吗？
- [ ] Full Ridge 的改善会不会来自 calendar 与 country dummy，而不是天气？
- [ ] 为什么没有报告 confidence interval 的跨 fold 稳定性？
- [ ] 如果明天加入 2025 数据后结果消失，你如何解释？
- [ ] 请说出一个会彻底推翻当前结论的证据。

## 二十四、现场计算与白板题

- [ ] 已知 persistence MAE 22.657、Full Ridge MAE 17.826，现场计算绝对和百分比改善。
- [ ] 已知 183 个每日 loss differences，写出 Newey-West t-stat 的计算框架。
- [ ] 写出 7 日 moving-block bootstrap 的伪代码。
- [ ] 解释为什么按 country-day IID bootstrap 会低估不确定性。
- [ ] 写出 Ridge 的目标函数，并解释 alpha 增大时偏差和方差如何变化。
- [ ] 写出 MAE、RMSE 和 R2 的定义。
- [ ] 构造一个 MAE 更好但 RMSE 更差的预测例子。
- [ ] 构造一个高 Pearson correlation 但 MAE 很差的例子。
- [ ] 画出一个 wind power curve，并标出 cut-in、rated 和 cut-out 区域。
- [ ] 画出日前拍卖、预测发布时间、决策时点和交付时点的时间线。
- [ ] 说明 23 小时和 25 小时 DST 日如何做加权日均。
- [ ] 给定三国当日 loss improvement，说明如何聚合成一个 daily observation。
- [ ] 设计 two-way clustered standard error 的思路。
- [ ] 设计一个 country-specific slope 的层级线性模型。
- [ ] 写出一个从 price forecast 转换为 bid quantity 的简化优化问题。

## 二十五、下一阶段研究计划

- [ ] 如果只有一周时间，优先修复哪三个问题？
- [ ] 如果有一个月时间，如何获得更长的 point-in-time forecast archive？
- [ ] 如何把国家质心替换为容量加权站点级天气？
- [ ] 如何引入实际 wind/solar generation forecast？
- [ ] 如何加入 load forecast、fuel、carbon、outages 和 interconnector flows？
- [ ] 如何从日均目标升级为 24 小时价格曲线？
- [ ] 如何做概率预测和 prediction interval？
- [ ] 如何设计 2025 至 2026 的真正 frozen holdout？
- [ ] 如何进行 shadow deployment 或 paper trading？
- [ ] 哪些验收条件满足后才可以声称具有业务价值？
- [ ] 哪些验收条件满足后才可以声称具有交易价值？
- [ ] 什么时候应该停止优化模型，转而改善数据？
- [ ] 什么时候才值得重新引入深度学习或图模型？
- [ ] 如果预算有限，最值得购买的外部数据是什么？
- [ ] 两周内你能交付的最可信增量是什么？

## 建议使用方式

1. 每个问题先准备 30 秒答案，再准备 2 分钟展开版本。
2. 所有数字都必须能定位到代码、CSV 或 JSON 报告。
3. 对没有完成的工作直接回答“尚未验证”，随后说明验证方案。
4. 区分预测显著性、经济价值和可交易性，不要混用。
5. 对负面结果主动解释，避免在追问后才承认。
6. 至少手算一次 MAE 改善、block bootstrap 和 Newey-West 的核心逻辑。
7. 面试前随机抽取 20 题，进行不看资料的模拟回答。
