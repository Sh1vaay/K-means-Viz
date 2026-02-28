<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5 Canvas" />
</p>

<h1 align="center">✨ ClusterViz ✨</h1>

<p align="center">
  <strong>An interactive, high-performance web application for learning, visualizing, and comparing machine learning clustering algorithms in real-time.</strong>
</p>

---

## 🔹 Project Information

* **Project Name:** ClusterViz
* **Short Description:** An interactive web application for learning, visualizing, and comparing machine learning clustering algorithms in real-time.
* **Problem It Solves:** Clustering algorithms inherently act as mathematical "black boxes." Visualizing how they group data, adjust centers, or identify outliers step-by-step is traditionally difficult. ClusterViz demystifies this process through interactive animations and comprehensive metrics.
* **Target Users:** Data science students, educators, and machine learning beginners seeking an intuitive understanding of clustering geometry.
* **Tech Stack:** Python, Flask, Scikit-Learn, SciPy, Vanilla JavaScript (ES6+), HTML5 Canvas, Chart.js, Plotly.js, CSS3.
* **Project Type:** Web Application / Educational Machine Learning Tool

---

## 📖 Project Overview and Purpose
ClusterViz is a visual playground for unsupervised machine learning. It allows users to generate datasets (like blobs, moons, and circles) and watch how various clustering algorithms (**K-Means, DBSCAN, Gaussian Mixture Models, and Hierarchical Clustering**) process the data step-by-step. The purpose is to provide an accessible, hands-on environment for intuitively grasping complex data science concepts without writing any code.

## 🎯 Problem Statement and Motivation
When learning machine learning, it can be hard to internalize what terms like *inertia*, *silhouette score*, or *epsilon* actually mean in practice. Standard tutorials output static images or text summaries. ClusterViz was built to bridge this gap, offering live animation and instant visual feedback so users can literally see the underlying math at work.

## ✨ Features and Functionality

| Feature | Description |
| :--- | :--- |
| 🧮 **Multi-Algorithm Support** | Visualize K-Means, DBSCAN, GMM, and Hierarchical Clustering. |
| 🎬 **Step-by-Step Animation** | Watch centroids move, Voronoi regions shift, and clusters form in real-time. |
| 📊 **Advanced Analytics Dashboard** | Deep-dive into metrics with an Elbow graph, Silhouette plots, PCA projection, Parallel Coordinates, Box Plots, and Feature Importance. |
| 🔀 **Comparison Mode** | Run all algorithms simultaneously side-by-side to see which performs best on the current dataset. |
| ↩️ **Time-Travel (Undo)** | Revert clustering steps instantly via a state history stack. |
| 🖌️ **Interactive Datasets & Tools** | Paint your own data points, drag centroids manually, or auto-suggest the best K value. |
| 🎨 **Themes & Palettes** | Switch between Dark (Cyberpunk) and Light themes, along with 4 distinct color palettes. |

## 🛠️ Tech Stack Explanation

### Backend
- **Python & Flask:** Chosen for its simplicity and lightweight nature. It effortlessly serves as an API layer to interface with Python's rich data science ecosystem.
- **Scikit-Learn & SciPy:** The industry standard for machine learning in Python. Chosen over writing algorithms from scratch to ensure absolute mathematical correctness and performance.

### Frontend
- **Vanilla JavaScript & HTML5 Canvas:** Selected over heavy frameworks (like React or Vue) to maximize rendering performance. Canvas allows drawing thousands of points smoothly at 60 FPS without DOM overhead.
- **Chart.js & Plotly.js:** Used for rendering the analytical charts (scatter plots, 3D views, bar graphs) because they provide beautiful, out-of-the-box interactive charts.

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.8+
- Git

### Step-by-Step Instructions
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd kmeanr
   ```
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you install `flask`, `scikit-learn`, `numpy`, and `scipy`)*

## 🚀 How to Run Locally

### ⚡ Fast Start (macOS & Linux only)
You can use the included bash script to automatically create your environment, install dependencies, and run the server in a single command:
```bash
./run.sh
```

### 🍎 macOS & 🐧 Linux (Manual)

1. **Activate your virtual environment:**
   ```bash
   source venv/bin/activate
   ```
2. **Run the Flask server:**
   ```bash
   python3 app.py
   ```
3. **Open a web browser** and navigate to `http://localhost:5000`.

### 🪟 Windows

1. **Activate your virtual environment:**
   ```cmd
   venv\Scripts\activate
   ```
   *(Note: If using PowerShell, run `.\venv\Scripts\Activate.ps1` instead)*
2. **Run the Flask server:**
   ```cmd
   python app.py
   ```
3. **Open a web browser** and navigate to `http://localhost:5000`.

---

## 📂 Folder Structure
```text
kmeanr/
├── app.py                 # Flask server and API route definitions
├── kmeans_logic.py        # Core ML logic, math, and data processing
├── static/
│   ├── css/
│   │   └── style.css      # Application styling and themes
│   └── js/
│       └── main.js        # Frontend state, Canvas rendering, and API calls
└── templates/
    └── index.html         # Main HTML view and layout structure
```

---

## 🔄 High-Level Workflow
1. **Data Generation:** The user selects a dataset shape on the frontend. Javascript makes a request to `/generate`. Python creates the data and returns it.
2. **Algorithm Execution:** When the user clicks "Start Animation", Javascript enters a polling loop. It repeatedly sends the current point data and cluster state to the backend (e.g., `/step`).
3. **Processing:** Python processes a single logical step of the algorithm (e.g., assigning points to nearest centroids) and returns the new state.
4. **Rendering:** Javascript updates its internal layout and commands the HTML Canvas to smoothly interpolate and draw the new positions, Voronoi cells, and colors.
5. **Evaluation:** Throughout the process, the frontend requests metrics from `/metrics`, updating the UI with Silhouette and Davies-Bouldin scores in real-time.

--
## ⚖️ License

**Copyright © 2026 Sh1vaay. All Rights Reserved.**

This repository is protected by a **Strict Proprietary License**. Any copying, reproduction, distribution, modification, or unauthorized use is strictly prohibited. For full details, please refer to the `LICENSE` file within this repository.
