# Breast Cancer Diagnosis App

## Summary

This project is a proof-of-concept diagnostic tool that uses **machine learning** and **DNA methylation data** to predict critical clinical outcomes for breast cancer patients — including:

- **Metastatic status**
- **Tumor stage**
- **Tumor location**
- **Likely metastasis site**

Built on real-world data from **The Cancer Genome Atlas (TCGA)** and leveraging genomic biomarkers, this app delivers fast, predictive insights that could one day assist physicians in making earlier and more personalized treatment decisions.

---

## What Makes It Different

Traditional diagnostics often rely on imaging and invasive procedures. This tool uses **non-invasive, epigenetic markers** — specifically **DNA methylation at 10 CpG sites** — to anticipate how a tumor will behave. These patterns are stable and can reveal metastasis risk before it becomes clinically evident.

The application bridges **genomic science**, **data mining**, and **predictive modeling**, aiming to support early detection and decision-making in oncology.

---

## Key Features

- **Input**: Upload a `.csv` or `.txt` file containing methylation data
- **Output**: Model predictions for:
  - Tumor stage (e.g., Stage II, Stage IV)
  - Whether cancer is metastatic
  - Location of tumor in breast tissue
  - Predicted metastasis site (e.g., bone, liver, lung)

- **Stack**: Python (Flask), JavaScript, HTML
- **Models**: Support Vector Machines, Random Forests, MultiOutput Classifiers

---

## Results (20-patient pilot)

| Prediction Task       | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| Metastatic Status      | 66.7%    | 83.3%     | 66.7%  | 66.7%    |
| Tumor Stage            | 57.1%    | 100%      | 57.1%  | 72.6%    |
| Tumor Subdivision      | ~76% avg | varies    | —      | —        |
| Metastasis Site        | 60%      | 70.8%     | 66.7%  | 65.1%    |

*MX patients (uncertain cases) were intentionally excluded from training to evaluate model performance in unknown scenarios.*

---

## Use Case

In a clinical setting, this app could be part of a pipeline where:
1. A patient’s blood is sequenced.
2. Methylation levels are extracted.
3. The app predicts how aggressive the cancer is and whether it's likely to spread.
4. A physician uses this to guide treatment strategy — earlier, faster, and more accurately.

---

## Vision

This project is a **prototype for next-generation diagnostics** — using the stability and predictive power of epigenetic data to enhance how we detect and treat cancer. With larger datasets and clinical partnerships, it can evolve into a platform that supports oncologists with AI-driven decision-making, grounded in genomic precision.

---

## Research Background

This work builds on prior research identifying hypermethylated CpG sites linked to metastatic breast cancer.  
**Full research paper available here**:  
[View Paper](https://www.overleaf.com/read/brgnjsjcgdgw#55609b)

---

## Contact

To collaborate, deploy, or explore this model in a clinical or enterprise setting, please reach out.

