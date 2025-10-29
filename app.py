"""
Streamlit Dashboard for Cancer Classification System

Interactive web application for:
- Uploading gene expression data
- Running Hybrid GA-PSO gene selection
- Training Transformer model
- Visualizing results and gene importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import io

# Import our modules
from data_preprocessing import DataPreprocessor, generate_synthetic_dataset
from optimization import HybridGAPSO
from transformer_model import GeneTransformerClassifier
from evaluation import ModelEvaluator
from interpretability import GeneImportanceAnalyzer

# Page configuration
st.set_page_config(
    page_title="Cancer Classification using GA-PSO & Transformer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode with high contrast
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #00D9FF;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #00FF9F;
    margin-top: 1.5rem;
    border-bottom: 2px solid #00FF9F;
    padding-bottom: 0.5rem;
}
.metric-card {
    background-color: #1E2127;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border: 1px solid #2E3137;
}

/* Enhanced contrast for readability */
.stMarkdown {
    color: #FAFAFA;
}

/* Improved button styling */
.stButton > button {
    background-color: #00D9FF;
    color: #0E1117;
    border: none;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #00FF9F;
    box-shadow: 0 0 15px rgba(0, 255, 159, 0.5);
}

/* Success message styling */
.stSuccess {
    background-color: rgba(0, 255, 159, 0.1);
    border-left: 4px solid #00FF9F;
}

/* Info message styling */
.stInfo {
    background-color: rgba(0, 217, 255, 0.1);
    border-left: 4px solid #00D9FF;
}

/* Metric value emphasis */
[data-testid="stMetricValue"] {
    color: #00D9FF;
    font-size: 1.8rem;
    font-weight: bold;
}

/* Sidebar improvements */
.css-1d391kg {
    background-color: #1E2127;
}

/* Code block styling */
code {
    background-color: #2E3137;
    color: #00FF9F;
    padding: 0.2rem 0.4rem;
    border-radius: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# Configure matplotlib for dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#1E2127'
plt.rcParams['axes.edgecolor'] = '#FAFAFA'
plt.rcParams['axes.labelcolor'] = '#FAFAFA'
plt.rcParams['text.color'] = '#FAFAFA'
plt.rcParams['xtick.color'] = '#FAFAFA'
plt.rcParams['ytick.color'] = '#FAFAFA'
plt.rcParams['grid.color'] = '#3E4147'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.facecolor'] = '#1E2127'
plt.rcParams['legend.edgecolor'] = '#3E4147'

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'genes_selected' not in st.session_state:
    st.session_state.genes_selected = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False


def main():
    # Title
    st.markdown('<p class="main-header">🧬 Intelligent Cancer Classification</p>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #A0A0A0; font-size: 1.1rem;">Using Hybrid GA-PSO Optimization and Transformer Networks</p>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Data loading section
    st.sidebar.header("1. Data Loading")
    
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Upload CSV File", "Generate Synthetic Data"]
    )
    
    data = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload gene expression CSV",
            type=['csv'],
            help="CSV file with genes as columns and 'label' column for classes"
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("Load Data"):
                with st.spinner("Loading and preprocessing data..."):
                    # Save uploaded file temporarily
                    with open("temp_data.csv", "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Preprocess
                    preprocessor = DataPreprocessor(random_state=42)
                    data = preprocessor.preprocess_pipeline("temp_data.csv", test_size=0.2)
                    
                    st.session_state.data = data
                    st.session_state.data_loaded = True
                    st.success("✅ Data loaded successfully!")
    
    else:  # Generate synthetic data
        st.sidebar.subheader("Synthetic Data Parameters")
        n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000, 100)
        n_genes = st.sidebar.slider("Number of genes", 1000, 20000, 5000, 1000)
        n_classes = st.sidebar.slider("Number of classes", 2, 10, 5, 1)
        
        if st.sidebar.button("Generate Data"):
            with st.spinner("Generating synthetic dataset..."):
                dataset_path = generate_synthetic_dataset(
                    n_samples=n_samples,
                    n_genes=n_genes,
                    n_classes=n_classes,
                    n_informative=200,
                    output_file="synthetic_data.csv"
                )
                
                preprocessor = DataPreprocessor(random_state=42)
                data = preprocessor.preprocess_pipeline(dataset_path, test_size=0.2)
                
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.success("✅ Synthetic data generated and loaded!")
    
    # Optimization section
    st.sidebar.header("2. Gene Selection (GA-PSO)")
    
    if st.session_state.data_loaded:
        st.sidebar.subheader("Optimization Parameters")
        
        n_particles = st.sidebar.slider("Population size", 10, 50, 30, 5)
        n_generations = st.sidebar.slider("Generations", 10, 100, 30, 10)
        n_selected_genes = st.sidebar.slider("Target genes", 50, 500, 150, 50)
        
        if st.sidebar.button("Run Gene Selection"):
            with st.spinner("Running Hybrid GA-PSO optimization..."):
                data = st.session_state.data
                
                optimizer = HybridGAPSO(
                    n_particles=n_particles,
                    n_generations=n_generations,
                    n_selected_genes=n_selected_genes,
                    random_state=42
                )
                
                results = optimizer.optimize(
                    data['X_train'], 
                    data['y_train'],
                    verbose=True
                )
                
                st.session_state.optimization_results = results
                st.session_state.genes_selected = True
                st.success(f"✅ Selected {len(results['selected_genes'])} genes!")
    
    # Model training section
    st.sidebar.header("3. Train Transformer")
    
    if st.session_state.genes_selected:
        st.sidebar.subheader("Model Parameters")
        
        embed_dim = st.sidebar.selectbox("Embedding dimension", [64, 128, 256], index=1)
        num_heads = st.sidebar.selectbox("Attention heads", [2, 4, 8], index=1)
        epochs = st.sidebar.slider("Training epochs", 10, 100, 50, 10)
        
        if st.sidebar.button("Train Model"):
            with st.spinner("Training Transformer model..."):
                data = st.session_state.data
                results = st.session_state.optimization_results
                selected_genes = results['selected_genes']
                
                # Select genes
                X_train_selected = data['X_train'][:, selected_genes]
                X_test_selected = data['X_test'][:, selected_genes]
                
                # Train model
                transformer = GeneTransformerClassifier(
                    n_genes=len(selected_genes),
                    n_classes=data['n_classes'],
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_transformer_blocks=2,
                    random_state=42
                )
                
                history = transformer.train(
                    X_train_selected, data['y_train'],
                    X_test_selected, data['y_test'],
                    epochs=epochs,
                    batch_size=32,
                    verbose=0
                )
                
                st.session_state.transformer = transformer
                st.session_state.history = history
                st.session_state.X_train_selected = X_train_selected
                st.session_state.X_test_selected = X_test_selected
                st.session_state.selected_genes = selected_genes
                st.session_state.model_trained = True
                st.success("✅ Model trained successfully!")
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👈 Please load data from the sidebar to get started.")
        
        # Show instructions
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Load Data**: Upload your gene expression CSV or generate synthetic data
        2. **Gene Selection**: Run Hybrid GA-PSO to select optimal genes
        3. **Train Model**: Train the Transformer neural network
        4. **View Results**: Analyze performance metrics and gene importance
        
        **Expected CSV Format:**
        - Rows: Samples/patients
        - Columns: Gene expression values
        - Required column: `label` (cancer class/type)
        """)
        
    elif st.session_state.data_loaded:
        # Display data information
        st.markdown('<p class="sub-header">📊 Data Overview</p>', unsafe_allow_html=True)
        
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Genes", data['n_features'])
        col2.metric("Training Samples", len(data['X_train']))
        col3.metric("Test Samples", len(data['X_test']))
        col4.metric("Classes", data['n_classes'])
        
        # Show class distribution
        st.markdown("#### Class Distribution")
        train_dist = pd.Series(data['y_train']).value_counts().sort_index()
        test_dist = pd.Series(data['y_test']).value_counts().sort_index()
        
        dist_df = pd.DataFrame({
            'Training': train_dist,
            'Testing': test_dist
        })
        st.bar_chart(dist_df)
        
    if st.session_state.genes_selected:
        # Display optimization results
        st.markdown('<p class="sub-header">🧬 Gene Selection Results</p>', unsafe_allow_html=True)
        
        results = st.session_state.optimization_results
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selected Genes", results['n_selected'])
        col2.metric("Best Fitness", f"{results['best_fitness']:.4f}")
        reduction = (1 - results['n_selected'] / st.session_state.data['n_features']) * 100
        col3.metric("Gene Reduction", f"{reduction:.1f}%")
        col4.metric("Optimization Time", f"{results['optimization_time']:.1f}s")
        
        # Plot fitness history with dark theme
        st.markdown("#### Optimization Progress")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(results['fitness_history'], label='Best Fitness', linewidth=3, color='#00D9FF')
        ax.plot(results['mean_fitness_history'], label='Mean Fitness', linewidth=3, color='#00FF9F', alpha=0.8)
        ax.set_xlabel('Generation', fontsize=12, color='#FAFAFA')
        ax.set_ylabel('Fitness', fontsize=12, color='#FAFAFA')
        ax.set_title('GA-PSO Convergence', fontsize=14, fontweight='bold', color='#00D9FF')
        ax.legend(facecolor='#1E2127', edgecolor='#3E4147', fontsize=10)
        ax.grid(alpha=0.3, color='#3E4147')
        st.pyplot(fig)
        plt.close()
        
        # Show selected genes
        with st.expander("View Selected Gene Indices"):
            selected_genes_list = results['selected_genes'].tolist()
            if 'feature_names' in st.session_state.data:
                feature_names = st.session_state.data['feature_names']
                selected_names = [feature_names[i] for i in selected_genes_list[:50]]
                st.write(f"First 50 selected genes: {', '.join(selected_names)}")
            else:
                st.write(f"First 50 genes: {selected_genes_list[:50]}")
    
    if st.session_state.model_trained:
        # Display model results
        st.markdown('<p class="sub-header">🤖 Model Performance</p>', unsafe_allow_html=True)
        
        transformer = st.session_state.transformer
        data = st.session_state.data
        X_test_selected = st.session_state.X_test_selected
        
        # Make predictions
        start_time = time.time()
        y_pred = transformer.predict(X_test_selected)
        y_pred_proba = transformer.predict_proba(X_test_selected)
        inference_time = time.time() - start_time
        
        # Evaluate
        evaluator = ModelEvaluator(n_classes=data['n_classes'])
        metrics = evaluator.evaluate(data['y_test'], y_pred, y_pred_proba)
        evaluator.compute_gene_reduction(
            data['n_features'],
            st.session_state.optimization_results['n_selected']
        )
        evaluator.add_timing(
            optimization_time=st.session_state.optimization_results['optimization_time'],
            training_time=transformer.training_time,
            inference_time=inference_time
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("F1-Score", f"{metrics['f1_macro']:.4f}")
        if metrics['roc_auc']:
            col3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        else:
            col3.metric("ROC-AUC", "N/A")
        col4.metric("Total Time", f"{evaluator.metrics['total_time']:.1f}s")
        
        # Training history
        st.markdown("#### Training History")
        history = st.session_state.history
        fig = evaluator.plot_training_history(history)
        st.pyplot(fig)
        plt.close()
        
        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        fig = evaluator.plot_confusion_matrix(data['y_test'], y_pred)
        st.pyplot(fig)
        plt.close()
        
        # ROC curves
        if metrics['roc_auc']:
            st.markdown("#### ROC Curves")
            fig = evaluator.plot_roc_curves(data['y_test'], y_pred_proba)
            st.pyplot(fig)
            plt.close()
        
        # Gene importance analysis
        st.markdown('<p class="sub-header">🔍 Gene Importance Analysis</p>', 
                   unsafe_allow_html=True)
        
        feature_names = [data['feature_names'][i] for i in st.session_state.selected_genes]
        analyzer = GeneImportanceAnalyzer(transformer.model, feature_names)
        
        top_k = st.slider("Number of top genes to display", 10, 50, 20, 5)
        
        with st.spinner("Computing gene importance..."):
            fig = analyzer.plot_top_genes(X_test_selected, top_k=top_k)
            st.pyplot(fig)
            plt.close()
        
        # Gene expression heatmap
        st.markdown("#### Gene Expression Heatmap")
        with st.spinner("Generating heatmap..."):
            fig = analyzer.plot_gene_heatmap(X_test_selected, data['y_test'], top_k=20)
            st.pyplot(fig)
            plt.close()
        
        # Summary report
        st.markdown("#### Evaluation Summary")
        report = evaluator.generate_summary_report()
        st.code(report, language=None)
        
        # Download results
        st.markdown("#### Download Results")
        
        # Create downloadable report
        report_buffer = io.StringIO()
        report_buffer.write("CANCER CLASSIFICATION RESULTS\n")
        report_buffer.write("="*60 + "\n\n")
        report_buffer.write(report)
        report_buffer.write("\nSelected Genes:\n")
        for i, gene_idx in enumerate(st.session_state.selected_genes[:50]):
            report_buffer.write(f"{i+1}. {feature_names[i]}\n")
        
        st.download_button(
            label="Download Report",
            data=report_buffer.getvalue(),
            file_name="cancer_classification_report.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
