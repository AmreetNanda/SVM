# Total bill prediction Regression App (Support Vector Machine + Streamlit + Docker)

This project demonstrates a fully modular SVM Regression implementation using Sklearn applied to the Tips dataset from seaborn
It includes:

- A complete machine learning pipeline
- A Streamlit web UI for calculating the total bill from different features 
- A fully modular ML codebase
- Optional Docker container for deployment

---

## Features

### Machine Learning
- Explanatory Data Analysis for the Total Bill prediction
- Custom SVM Regression sklearn model
- Automatic feature normalization
- Training metrics and RMSE visualization

## Project Structure
```bash
SVM/
│
├── app.py # Streamlit UI
│
├── Support_Vector_Machine/ # Modular ML package
│ ├── init.py
│ ├── data_utils.py
│ ├── inference.py
│ ├── preprocess.py
│ ├── models.py
│ ├── main.py
│
├── requirements.txt # Python dependencies
└── Dockerfile # Docker container
└── run.sh # Optional to run the script
```

## Installation

## 🛠 Installation (without Docker)

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/SVM.git
cd Support_Vector_Machine
```

### 2. Install dependencies
```bash
pip install -r Requirements.txt
```

### 3. Run main file to train the model and create the pkl files
```bash
python main.py
```

### 4. Run Streamlit
```bash
streamlit run app.py
```
Open in your browser:
👉 http://localhost:8501

## 🐳 Running with Docker (optional)
### Build the image
```bash
docker build -t support_vector_machine .
```

### Run the container
```bash
docker run -p 8501:8501 support_vector_machine
```
Open: 👉 http://localhost:8501
## Screenshots

![App Screenshot](https://github.com/AmreetNanda/SVM/blob/main/SVM_png.png)


## License

[MIT](https://choosealicense.com/licenses/mit/)

