"""
Transformer-Based Cancer Classification Model

This module implements a Transformer neural network for cancer classification using
selected gene expression features.

Architecture:
- Embedding layer for gene expression
- Multi-head self-attention mechanism
- Feed-forward neural network
- Classification head with softmax output
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import warnings
warnings.filterwarnings('ignore')


class TransformerBlock(layers.Layer):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    
    Components:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        """
        Initialize Transformer block.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Hidden dimension of feed-forward network
            dropout_rate (float): Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        """
        Forward pass through the Transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor after attention and feed-forward operations
        """
        # Multi-head self-attention with residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class GeneTransformerClassifier:
    """
    Transformer-based model for cancer classification from gene expression data.
    
    Architecture:
    1. Input: Gene expression values (selected features)
    2. Embedding: Project genes to higher dimension
    3. Positional encoding: Add position information
    4. Transformer blocks: Multi-head attention and FFN
    5. Global pooling: Aggregate gene representations
    6. Dense layers: Final classification
    7. Output: Softmax probabilities over cancer classes
    """
    
    def __init__(self,
                 n_genes,
                 n_classes,
                 embed_dim=128,
                 num_heads=4,
                 ff_dim=256,
                 num_transformer_blocks=2,
                 mlp_units=[128, 64],
                 dropout_rate=0.1,
                 learning_rate=0.001,
                 random_state=42):
        """
        Initialize Transformer classifier.
        
        Args:
            n_genes (int): Number of input genes (features)
            n_classes (int): Number of output classes
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward network hidden dimension
            num_transformer_blocks (int): Number of Transformer blocks
            mlp_units (list): Hidden units in final MLP
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random seed
        """
        self.n_genes = n_genes
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.history = None
        self.training_time = 0
    
    def build_model(self):
        """
        Build the Transformer model architecture.
        
        Returns:
            Compiled Keras model
        """
        print("Building Transformer model...")
        
        # Input layer
        inputs = layers.Input(shape=(self.n_genes,))
        
        # Reshape for sequence processing: (batch, genes, 1)
        x = layers.Reshape((self.n_genes, 1))(inputs)
        
        # Embedding layer: project each gene to embed_dim
        x = layers.Dense(self.embed_dim)(x)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=self.n_genes, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.n_genes, 
            output_dim=self.embed_dim
        )(positions)
        x = x + position_embedding
        
        # Stack Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(x)
        
        # Global average pooling to aggregate gene information
        x = layers.GlobalAveragePooling1D()(x)
        
        # MLP head for classification
        for units in self.mlp_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with softmax
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nModel architecture:")
        print(f"  Input genes: {self.n_genes}")
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  Attention heads: {self.num_heads}")
        print(f"  Transformer blocks: {self.num_transformer_blocks}")
        print(f"  Output classes: {self.n_classes}")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, verbose=1):
        """
        Train the Transformer model.
        
        Args:
            X_train (array): Training features (selected genes)
            y_train (array): Training labels
            X_val (array, optional): Validation features
            y_val (array, optional): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
            
        Returns:
            History object with training metrics
        """
        print("\n" + "="*60)
        print("TRAINING TRANSFORMER MODEL")
        print("="*60 + "\n")
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"Training with validation set")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"Training configuration:")
        print(f"  Training samples: {len(X_train)}")
        if validation_data:
            print(f"  Validation samples: {len(X_val)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {epochs}")
        print()
        
        # Train model
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\nTraining completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array): Input features
            
        Returns:
            array: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (array): Input features
            
        Returns:
            array: Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import generate_synthetic_dataset, DataPreprocessor
    from optimization import HybridGAPSO
    
    print("Generating synthetic dataset...")
    dataset_path = generate_synthetic_dataset(
        n_samples=500,
        n_genes=1000,
        n_classes=3
    )
    
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(dataset_path)
    
    print("\nRunning gene selection (quick test)...")
    optimizer = HybridGAPSO(
        n_particles=10,
        n_generations=5,
        n_selected_genes=50
    )
    results = optimizer.optimize(data['X_train'], data['y_train'], verbose=False)
    
    # Select genes
    selected_genes = results['selected_genes']
    X_train_selected = data['X_train'][:, selected_genes]
    X_test_selected = data['X_test'][:, selected_genes]
    
    print(f"\nTraining Transformer with {len(selected_genes)} selected genes...")
    transformer = GeneTransformerClassifier(
        n_genes=len(selected_genes),
        n_classes=data['n_classes'],
        embed_dim=64,
        num_heads=2,
        num_transformer_blocks=1,
        random_state=42
    )
    
    transformer.train(
        X_train_selected, data['y_train'],
        X_test_selected, data['y_test'],
        epochs=20,
        batch_size=32
    )
    
    # Make predictions
    predictions = transformer.predict(X_test_selected)
    accuracy = (predictions == data['y_test']).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
