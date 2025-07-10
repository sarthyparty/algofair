# Student Dataset Fairness Analysis

Intersectional fairness analysis using GerryFair and baseline approaches on student performance data.

## 🚀 Quick Start

```bash
python main_analysis.py
```

Generates all publication-ready figures and statistical analysis in the `outputs/` folder.

## 📊 Generated Figures

### **Figure 1: FPR Bias Analysis** 
**Significance**: Reveals systematic bias against male students (-0.155 FPR difference), with minimal location-based bias.

### **Figure 2: Group Protection During Training**
**Significance**: Shows GerryFair learns to prioritize Female-Urban students (65% protection) over Male-Rural students (8%), indicating intersectional vulnerabilities.

### **Figure 3: Vulnerable Students Focus**
**Significance**: Among failing students, Female-Urban protection reaches 80%, showing fairness interventions are critical for intersectional identities.

## 🔬 Dataset

- **Source**: Portuguese student performance (395 students)
- **Target**: G3 final grade → binary pass/fail
- **Protected Attributes**: Gender (Female/Male), Location (Urban/Rural), Intersectional combinations

## 🎯 Key Findings

1. **Baseline Bias**: Males privileged over females (-0.155 FPR difference)
2. **Intersectional Effects**: Female-Urban students need most protection
3. **Vulnerable Focus**: Protection amplified for failing intersectional groups
4. **Algorithm Learning**: GerryFair successfully detects vulnerable demographics

## 📁 Structure

```
├── main_analysis.py              # Main script
├── clean_data/student_data.csv   # Dataset  
├── outputs/                      # Results (figures + stats)
├── functions/ & gerryfair/       # Core code
└── archive/                      # Old experiments (git ignored)
```
