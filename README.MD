# 🧠 AI for Injection Molding: Digital Twin Pipeline ⚙️

An intelligent Machine Learning (ML) pipeline built to simulate a **Digital Twin** for the injection molding manufacturing process. 

In real-world manufacturing, inspecting injected parts manually is too slow for real-time quality control. This project showcases how deep learning can instantly predict injected part quality (**Part Weight**) based on real-time machine processing conditions, allowing operators or automated systems to adjust parameters on the fly.

---

## 📊 How the ML Model Works

The model maps the complex relationship between **4 core machine processing inputs** to predict a critical quality metric:

* **📥 Inputs (Features):**
    * `InjectionPressure` (bar)
    * `MeltTemp` (°C)
    * `CoolingTime` (seconds)
    * `PackingPressure` (bar)
* **📤 Output (Target):**
    * `PartWeight` (grams) — *Predicted vs. Actual*

---

## 🚀 Quick Start & Workflow

Follow these steps to run the Digital Twin pipeline locally on your Linux machine:

### Step 1: Clone the Repository
```bash
git clone [https://github.com/PongKool/AI-for-Injection-Molding.git](https://github.com/PongKool/AI-for-Injection-Molding.git)
cd AI-for-Injection-Molding
