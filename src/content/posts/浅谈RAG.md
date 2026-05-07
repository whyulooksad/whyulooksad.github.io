---
title: 浅谈RAG
date: 2026-04-15 11:34:04
tags: 
---

今天来谈谈agent工程中殿堂级的技术栈——RAG。今天主要是

今天来谈谈RAG实战中的一些细节。

# RAG介绍

## 1. 什么是RAG?

RAG（Retrieval-Augmented Generation）是一种结合了 **信息检索（Retrieval）** 和 **文本生成（Generation）**的AI技术框架。它的核心思想是，在生成文本时，不仅依赖于模型本身的参数，还可以从外部知识库中检索相关的信息，以增强生成的内容。

## **2. RAG 的工作过程**

RAG分为离线和在线两个部分。

离线部分是指系统的准备阶段，主要完成**知识库的建立与向量化**。

在线部分主要包括两个阶段：

1. **检索（Retrieval）**：从知识库（如文档、数据库、互联网等）中检索出与输入问题最相关的内容。
2. **生成（Generation）**：利用检索到的信息作为额外的上下文输入，引导大语言模型（如GPT）生成更准确、更可靠的答案。

一个基本的RAG流程如下：

![image-20260416000747617](https://cdn.jsdelivr.net/gh/whyulooksad/image_bed@main/images/20260416000747927.png)

# 解析

首先是知识文档库中文档的解析，因为大多都是PDF，所以这里我直接以PDF为例。

在总体方案的选型上，我尝试过PyPDF2、Unstructured、Deepdoc。其中，Deepdoc的效果整体来说最好

| PyPDF2          | 只能提取文本，无法处理复杂布局                               |
| --------------- | ------------------------------------------------------------ |
| **Unstructure** | **通用性强但复杂表格识别率一般**                             |
| **Deepdoc**     | **基于YOLOv10的布局分析+PaddleOCR，对中文文档友好，图表识别率较高** |

PDF的 pipeline：

```
输入PDF → 版面分析 → 元素分类 → 针对性解析 → 结构化输出
```

**step1: 版面分析**

一般情况下，模型直接采用 Deepdoc 默认的 YOLOv10 即可。我也看到有人会改用 LayoutLMv3，但我没有试过。

该方案通常会识别七类版面元素：标题、正文、表格、图片、页眉页脚、印章和水印，并输出每个元素的 bbox 坐标、类别以及置信度。



这里补充一个合同场景中很常见的问题：印章或签名经常会与正文发生重叠，导致 OCR 识别效果下降。我的处理思路是增加一个遮挡检查模块：当某个 text 块的 OCR 置信度低于 0.6，且与印章区域的 IoU 大于 0.3 时，先执行印章去除，再重新进行 OCR。具体做法上，可以用 `color histogram + connected component analysis` 来分离印章层和文本层，从而尽量减少遮挡对识别结果的影响。

**step2：表格解析**

文档中通常有有边框表格和无边框表格两类表格。有边框表格deepdoc默认可以处理，而通过空格对齐的无边框表格，deepdoc的识别率一般。

我给出的方案是先用启发式规则判断有框表格和无框表格。如果是无框表格，调用MinerU 2.5来输出HTML结构化表格。

MinerU 2.5原理：基于Table Transformer架构，用row/column detection识别隐式表格结构，然后用cell matching重建单元格关系。

**step3：扫描件/模糊图片处理**

有些PDF是做了双层的，或者里面直接贴了很多扫描件，针对这种模糊场景，我的优化思路是：先用拉普拉斯方差评估扫描图像的模糊程度，再做分级处理：清晰图直接进入解析流程；中度模糊图依次进行去噪、锐化、超分辨率和对比度增强；严重模糊图则直接切换到预处理能力更强的 MinerU 2.5，以提升整体识别鲁棒性。

```
def preprocess_scanned_image(img):
    # Step 1: 模糊检测（拉普拉斯方差）

    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()

    if blur_score > 100:  # 清晰图片
        return img

    elif 50 < blur_score <= 100:  # 中度模糊
        # 去噪 → 锐化 → 超分辨率 → 对比度增强
        img = cv2.fastNlMeansDenoising(img)
        img = cv2.filter2D(img, -1, sharpen_kernel)
        img = sr_model.predict(img)  # Real-ESRGAN
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=10)

    else:  # 严重模糊（blur_score <= 50）
        # 直接用MinerU 2.5（它内置了更强的预处理）
        return minerU_process(img)

    return img
```

**step4：非文本元素处理**

对于 PDF 中的图片和公式，我会在解析阶段先做类型识别，再分别转成文本化表示。流程图、示意图这类语义图片，使用多模态模型生成摘要描述；数据图表使用 chart-to-text 模型提取数据；数学公式则通过 LaTeX-OCR 转成 文本表达。这样做就可以把原本不可检索的非文本信息转换成可入库、可召回的结构化内容，再在分块阶段挂到对应 chunk 中。

# 分块
