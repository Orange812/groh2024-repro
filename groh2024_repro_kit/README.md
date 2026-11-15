# Groh et al. 2024 复现与演示工具包

论文：Groh M, Badri O, Daneshjou R, et al. *Deep learning-aided decision support for diagnosis of skin disease across skin tones.* **Nature Medicine** 30, 573–583 (2024). doi:10.1038/s41591-023-02728-3

> 这是为课程演讲编写的**独立复现代码**，不是论文作者发布的官方代码。官方代码与数据见下方"作者公开资源"。

## 一、作者公开资源（演讲里要介绍的"源代码 + 数据集"）

| 资源 | 位置 | 内容 | 获取方式 |
|---|---|---|---|
| 实验数据 + 分析代码 + AI 训练代码 | ResearchBox #1802：https://researchbox.org/1802 （Zenodo 备份 doi:10.5281/zenodo.15039588） | `Experimental Data.csv.zip`（1.2 MB，全部医生诊断记录）、`Analysis/Analysis.ipynb`（复现论文图表）、`Analysis/requirements.txt`、`Control DLS Training Code.py`（对照 AI 训练代码） | 网页上点 *entire box* 直接下载，CC BY 4.0 |
| 实验用的 364 张图片 | Zenodo：doi:10.5281/zenodo.10070478 | 46 种皮肤病图片 | 受限访问：需注册 Zenodo、同意"仅科研/医学用途、用完删除"，**提前几天申请** |
| AI 训练用的公开数据集 | https://github.com/mattgroh/fitzpatrick17k | 16,577 张临床图片 + Fitzpatrick 分型标注（`fitzpatrick17k.csv`）、`train.py` | CSV 直接下载；图片按 url 列下载或填表申请，CC BY-NC-SA 3.0 |
| 原实验网站 | https://diagnosing-diagnosis.media.mit.edu | 论文的在线实验平台 | 演讲前自己点开确认是否仍可访问 |

## 二、本工具包包含什么

```
groh2024_repro_kit/
├─ 00_check_env.py            检查环境
├─ run_demo.bat / run_demo.sh 一键：模拟数据 → 分析 → 启动演示网站
├─ requirements.txt           分析 + 网站依赖（pandas numpy scipy matplotlib flask）
├─ requirements-dls.txt       AI 训练依赖（torch torchvision，建议用 Colab）
├─ common/diseases.py         46 种疾病名、同义词、自由作答判分（拼写容错）
├─ analysis/
│   ├─ inspect_data.py        第一步：查看作者数据的列名，自动生成列名映射 mapping.json
│   ├─ key_results.py         独立复现核心结果（8 张表 + 3 张图，并列"论文报告值"）
│   ├─ stats_utils.py         OLS + 参与者聚类稳健标准误（与论文方法一致，无需 statsmodels）
│   └─ make_synthetic_data.py 生成模拟数据（彩排/测试用，绝不能当论文结果）
├─ dls/
│   ├─ train_control_dls.py   VGG-16 9 分类对照 AI（按论文 Methods 重写）
│   ├─ wizard_of_oz.py        增强 AI：错误预测随机改对 65%（47% → 约 84%）
│   ├─ predict.py             对图片推理 / 给演示网站批量写入 AI 建议
│   ├─ fetch_f17k_images.py   下载 Fitzpatrick17k 图片（demo 模式 / training 模式）
│   └─ colab_train_demo.ipynb 在 Colab GPU 上 30–60 分钟训练演示模型
└─ demo_app/                  论文实验平台的复刻版（Flask）
    ├─ app.py                 问卷 → 看图诊断 → AI 建议（按钮顺序随机）→ 反馈 → 对比 → 实时汇总
    ├─ import_images.py       导入你有权使用的图片（如 Zenodo 实验图）
    └─ data/images.csv        图片清单（默认 30 张占位图，换成真实图片见下文）
```

## 三、5 分钟快速开始（不需要任何下载）

```bash
pip install -r requirements.txt
python 00_check_env.py
python analysis/make_synthetic_data.py                     # 生成模拟数据
python analysis/key_results.py --data data/synthetic_experiment.csv
python demo_app/app.py                                     # 浏览器打开 http://127.0.0.1:5000
```
Windows 也可以直接双击 `run_demo.bat`。模拟数据的输出都带有 "SIMULATED DATA" 标记。

## 四、用作者真实数据复现

**路径 A：运行作者的官方 notebook（最权威，演讲中优先展示）**
```bash
# 解压 ResearchBox_1802.zip 后
cd Analysis
pip install -r requirements.txt     # 作者给的固定版本较旧，若安装失败用下面的兼容环境
python -m notebook                  # 打开 Analysis.ipynb，Run All
```
兼容环境（推荐的版本组合，尚未实际测试；如有报错按提示微调版本）：
```bash
conda create -n groh python=3.9 pandas=1.4 numpy=1.21 scipy=1.7 matplotlib=3.5 seaborn=0.11 statsmodels=0.13 notebook -c conda-forge
conda activate groh && pip install stargazer
```
如果 notebook 第一个单元格的数据路径与你解压的位置不同，改成 `Experimental Data.csv.zip` 的实际路径即可。

**路径 B：用本工具包独立复现（展示"复现值 vs 论文值"）**
```bash
python analysis/inspect_data.py "Experimental Data.csv.zip" --write-map mapping.json
# 打开 mapping.json 核对列名（拿不准就在 Analysis.ipynb 里搜这个列名看作者怎么用）
python analysis/key_results.py --data "Experimental Data.csv.zip" --map mapping.json --out outputs/real
```
输出：`outputs/real/results_summary.md`（所有表格）、`表1.csv`…、`fig1/2/3*.png`。
注意：回归设定是按论文文字推断的，个别系数定义可能与原文略有差异；以官方 notebook 为准。

## 五、AI 模型（可选）

- 在 Colab 打开 `dls/colab_train_demo.ipynb`，运行时选 GPU，按单元格执行，最后下载 `checkpoint.zip`，解压到本地 `dls/checkpoint/`。
- 本地推理：`python dls/predict.py --ckpt dls/checkpoint 图片.jpg`
- 增强 AI 原理演示（无需模型）：`python dls/wizard_of_oz.py --demo`
- 论文的对照 AI 还用了作者私有图片，公开数据只能训练出近似模型，准确率不会与论文一致。

## 六、演示网站换成真实图片

- 公开图片（推荐课堂使用）：下载 `fitzpatrick17k.csv` 后运行
  `python dls/fetch_f17k_images.py --csv fitzpatrick17k.csv --per-class 4 --n-other 6`
- 你已获授权的 Zenodo 实验图：`python demo_app/import_images.py --src 图片目录 --labels 标签.csv --file-col ... --disease-col ... --fst-col ...`
  （这些图片只能在你自己的电脑上本地展示，不能上传到公开网站或分享。）
- 有训练好的模型时，`python dls/predict.py --update-demo` 会把真实 AI 预测写进清单；否则网站按 47% 准确率模拟 AI 建议（同一张图对所有人给同样的建议，与论文一致）。

## 七、课堂演示小贴士

- 固定分组演示两种界面：`/?dls=control&ui=control` 和 `/?dls=control&ui=treatment`
- 同学用手机参与：`python demo_app/app.py --lan`，把电脑局域网 IP 告诉大家（校园网可能禁止设备互访，提前测试）
- `/dashboard` 每 10 秒刷新全班结果；`/export.csv` 下载数据，再用
  `python analysis/key_results.py --data classroom_responses.csv --min-trials 1 --keep-failed-attention` 当场分析
- "Start new session" 会把旧记录改名备份，不会删除

## 八、常见问题

- **ModuleNotFoundError**：在工具包根目录运行命令，并先 `pip install -r requirements.txt`。
- **端口被占用**：`python demo_app/app.py --port 5050`。
- **中文乱码（Windows）**：用 `run_demo.bat`，或先执行 `chcp 65001`。
- **图片下载大量失败**：原图谱网站链接失效，按 Fitzpatrick17k 仓库 README 填表向作者申请整包图片。

## 许可与引用

作者的 ResearchBox 内容为 CC BY 4.0；Fitzpatrick17k 为 CC BY-NC-SA 3.0（非商业、需署名）；Zenodo 实验图片受访问条款约束。使用时请引用原论文与 ResearchBox #1802。
