# 🧵 ThreadScope – String Art Quality Analyzer

ThreadScope is a Python tool that **simulates string art from nail-sequence files and measures how closely the result matches an original image**.

It helps artists and developers objectively evaluate string art algorithms by generating:

- 📈 Similarity vs Thread Count graph  
- 🖼 Side-by-side visual comparison  
- 📋 Detailed simulation logs  

The similarity is calculated using the **Structural Similarity Index (SSIM)**.

---

# 📷 Example Output

## Similarity vs Thread Count

![Similarity Graph](images/similarity_graph.png)

This graph shows how increasing thread count affects the similarity between the simulated string art and the original image.

---

## Visual Comparison Strip

![Comparison Strip](images/comparison_strip.png)

Displays:

- Original image  
- Simulated string art at different thread counts  
- Similarity percentage for each version  

---

# 🚀 Features

## Image Processing

- Converts images to grayscale
- Resizes images to a circular frame
- Masks areas outside the circular art region

## String Art Simulation

- Simulates thread paths between nails
- Accumulates brightness to simulate overlapping threads
- Applies blur and gamma correction for realistic thread appearance

## Quality Evaluation

- Uses **SSIM (Structural Similarity Index)**
- Tests both **normal and inverted image comparisons**
- Automatically chooses the best similarity score

## Visualization

- Interactive GUI built with Tkinter
- Graph plotting using Matplotlib
- Image processing with Pillow

---

# 🧠 How It Works

## 1. Input Image

The selected image is:

- Converted to grayscale  
- Resized to a square canvas  
- Masked into a circular frame  

This represents the physical string art board.

---

## 2. Thread Sequence Files

Each `.txt` file contains a sequence of nail indices representing the path of the thread.

Example sequence file:

```
12,45,200,34,11,98
```

Each pair of indices represents a thread segment between two nails.

---

## 3. Thread Simulation

The system simulates thread placement by:

1. Connecting nail coordinates
2. Accumulating brightness where threads overlap
3. Applying blur to simulate thread thickness

---

## 4. Similarity Measurement

The simulated string art is compared with the original image using **SSIM**.

Similarity scale:

```
0%   = completely different
100% = identical image
```

---

# 🖥 Interface

The application interface includes:

### Inputs
- Source Image
- Thread Sequence Files

### Adjustable Settings
- Number of Nails
- Image Size
- Thread Opacity

### Output Tabs
- 📈 Similarity Graph
- 🖼 Comparison Strip
- 📋 Log Output

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Kevin-Davees/ThreadScope-String-Art-Quality-Analyzer.git
cd threadscope
```

Install dependencies:

```bash
pip install pillow scikit-image matplotlib numpy
```

---

# ▶ Running the Program

Run the analyzer with:

```bash
python string_art_analyzer.py
```

Steps:

1. Select a source image
2. Select one or more thread sequence `.txt` files
3. Adjust settings if needed
4. Click **Run Analysis**

---

# 📁 Project Structure

```
threadscope/
│
├── string_art_analyzer.py
│
├── images/
│   ├── similarity_graph.png
│   ├── comparison_strip.png
│   └── ui_preview.png
│
├── examples/
│   ├── 800.txt
│   ├── 1200.txt
│   └── 1600.txt
│
└── README.md
```

---

# 🧪 Example Workflow

1. Generate string art sequences using your preferred algorithm.

2. Save sequence files such as:

```
800.txt
1200.txt
1600.txt
```

3. Run the analyzer and load these files.

4. The program will generate:

- A similarity graph
- A stitched comparison image
- Detailed logs

Output image:

```
string_art_comparison.png
```

This file contains the original image and all simulated thread art images.

---

# 📊 Example Results

| Threads | Similarity |
|--------|-----------|
| 800 | 34% |
| 1200 | 49% |
| 1600 | 63% |
| 2000 | 71% |

This shows how increasing thread density improves visual accuracy.

---

# 🛠 Requirements

- Python 3.8+
- NumPy
- Pillow
- Matplotlib
- scikit-image
- Tkinter

Install them using:

```bash
pip install pillow scikit-image matplotlib numpy
```

---

# 📌 Applications

ThreadScope can be used for:

- Evaluating string art algorithms
- Finding the optimal thread count
- Comparing different nail routing strategies
- Research experiments in computational art
- Improving automated string art generation systems

---

# 🧩 Future Improvements

Potential enhancements:

- GPU-accelerated simulation
- Automatic nail placement optimization
- Batch processing of hundreds of sequences
- Exportable similarity reports
- Integration with string art generators

---

# 👤 Author

Peter
