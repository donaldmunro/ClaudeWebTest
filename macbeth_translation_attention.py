"""
Attention Mechanism for English-to-French Translation
Demonstrates the difference between Self-Attention and Cross-Attention

Key Difference:
- Self-Attention: Q, K, V all come from the SAME sequence
- Cross-Attention: Q comes from target (French), K and V come from source (English)
"""

import numpy as np
from typing import Tuple

# Macbeth's Soliloquy
ENGLISH_TEXT = """Tomorrow, and tomorrow, and tomorrow,
creeps in this petty pace from day to day,
to the last syllable of recorded time;
and all our yesterdays have lighted fools
the way to a dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player,
that struts and frets his hour upon the stage,
and then is heard no more. It is a tale
told by an idiot, full of sound and fury,
signifying nothing."""

FRENCH_TEXT = """Demain, et demain, et demain,
rampe à ce rythme mesquin de jour en jour,
jusqu'à la dernière syllabe du temps écrit;
et tous nos hiers ont éclairé des fous
sur le chemin d'une mort poussiéreuse. Éteins-toi, éteins-toi, brève chandelle!
La vie n'est qu'une ombre qui marche, un pauvre acteur,
qui se pavane et s'agite son heure sur la scène,
et qu'ensuite on n'entend plus. C'est un récit
conté par un idiot, plein de bruit et de fureur,
ne signifiant rien."""


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray,
                   W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-Attention: Used within encoder or decoder

    KEY POINT: Q, K, and V all derived from the SAME input sequence X

    Args:
        X: Input sequence (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices for Q, K, V projections

    Returns:
        output: Attention output
        attention_weights: Attention score matrix
    """
    # All come from the SAME sequence
    Q = X @ W_q  # Query from X
    K = X @ W_k  # Key from X (SAME as Q source)
    V = X @ W_v  # Value from X (SAME as Q source)

    d_k = K.shape[-1]

    # Attention scores
    scores = (Q @ K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = attention_weights @ V

    return output, attention_weights


def cross_attention(X_target: np.ndarray, X_source: np.ndarray,
                    W_q: np.ndarray, W_k: np.ndarray,
                    W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-Attention (Encoder-Decoder Attention): Used for translation

    KEY POINT: Q from target (French), but K and V from source (English)
    This is the CRITICAL DIFFERENCE from self-attention!

    Args:
        X_target: Target sequence embeddings (French) - shape: (target_len, d_model)
        X_source: Source sequence embeddings (English) - shape: (source_len, d_model)
        W_q: Query weight matrix
        W_k: Key weight matrix
        W_v: Value weight matrix

    Returns:
        output: Attention output
        attention_weights: Attention score matrix showing which source words
                          each target word attends to
    """
    # CRITICAL DIFFERENCE FROM SELF-ATTENTION:
    Q = X_target @ W_q   # Query from TARGET (French decoder state)
    K = X_source @ W_k   # Key from SOURCE (English encoder output) ← DIFFERENT!
    V = X_source @ W_v   # Value from SOURCE (English encoder output) ← DIFFERENT!

    d_k = K.shape[-1]

    # Attention scores: each French word attending to English words
    scores = (Q @ K.T) / np.sqrt(d_k)  # Shape: (target_len, source_len)
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of source values
    output = attention_weights @ V

    return output, attention_weights


def create_mock_embeddings(text: str, d_model: int = 64) -> np.ndarray:
    """Create simple mock embeddings for demonstration."""
    words = text.lower().split()
    # Simple hash-based embedding for demo
    np.random.seed(42)
    embeddings = []
    for word in words:
        seed = sum(ord(c) for c in word)
        np.random.seed(seed)
        embeddings.append(np.random.randn(d_model))
    return np.array(embeddings)


def visualize_attention(attention_weights: np.ndarray,
                       source_words: list,
                       target_words: list,
                       top_k: int = 3) -> None:
    """Visualize which source words each target word attends to."""
    print("\n" + "="*80)
    print("CROSS-ATTENTION VISUALIZATION: French → English Alignment")
    print("="*80)
    print(f"Showing top {top_k} attended English words for each French word:\n")

    for i, target_word in enumerate(target_words[:10]):  # Show first 10 for brevity
        top_indices = np.argsort(attention_weights[i])[-top_k:][::-1]
        top_scores = attention_weights[i][top_indices]

        print(f"French word: '{target_word}'")
        print(f"  Attends to English words:")
        for idx, score in zip(top_indices, top_scores):
            if idx < len(source_words):
                print(f"    - '{source_words[idx]}' (weight: {score:.4f})")
        print()


def main():
    print("="*80)
    print("ATTENTION MECHANISM FOR TRANSLATION: English → French")
    print("="*80)
    print("\n📖 MACBETH'S SOLILOQUY\n")
    print("English (Source):")
    print("-" * 40)
    print(ENGLISH_TEXT)
    print("\nFrench (Target):")
    print("-" * 40)
    print(FRENCH_TEXT)

    # Setup
    d_model = 64
    d_k = 32

    # Create mock embeddings
    english_words = ENGLISH_TEXT.lower().split()
    french_words = FRENCH_TEXT.lower().split()

    X_english = create_mock_embeddings(ENGLISH_TEXT, d_model)
    X_french = create_mock_embeddings(FRENCH_TEXT, d_model)

    # Initialize weight matrices
    np.random.seed(42)
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1
    W_v = np.random.randn(d_model, d_k) * 0.1

    print("\n" + "="*80)
    print("SELF-ATTENTION vs CROSS-ATTENTION: KEY DIFFERENCES")
    print("="*80)

    print("\n1️⃣  SELF-ATTENTION (used in encoder/decoder internally)")
    print("-" * 40)
    print("Purpose: Allow each word to attend to other words in the SAME sequence")
    print("\nMatrix Sources:")
    print("  • Q (Query):  Derived from X_english  ← Same source")
    print("  • K (Key):    Derived from X_english  ← Same source")
    print("  • V (Value):  Derived from X_english  ← Same source")
    print("\nUse case: Understanding context within English text")

    self_output, self_attn = self_attention(X_english, W_q, W_k, W_v)
    print(f"\nOutput shape: {self_output.shape}")
    print(f"Attention weights shape: {self_attn.shape} (English → English)")

    print("\n2️⃣  CROSS-ATTENTION (used for translation)")
    print("-" * 40)
    print("Purpose: Allow French words to attend to relevant English words")
    print("\n⚠️  CRITICAL DIFFERENCE - Matrix Sources:")
    print("  • Q (Query):  Derived from X_french   ← Target (what we're generating)")
    print("  • K (Key):    Derived from X_english  ← Source (DIFFERENT!)")
    print("  • V (Value):  Derived from X_english  ← Source (DIFFERENT!)")
    print("\nUse case: Translation - mapping French words to English context")

    cross_output, cross_attn = cross_attention(X_french, X_english, W_q, W_k, W_v)
    print(f"\nOutput shape: {cross_output.shape}")
    print(f"Attention weights shape: {cross_attn.shape} (French → English)")

    # Visualize some attention patterns
    visualize_attention(cross_attn, english_words, french_words, top_k=3)

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
In SELF-ATTENTION:
  - Q and K come from the SAME sequence
  - Example: "tomorrow" attending to "tomorrow", "and", "creeps" (all English)
  - Attention matrix is square: (seq_len × seq_len)

In CROSS-ATTENTION for Translation:
  - Q comes from TARGET sequence (French decoder)
  - K and V come from SOURCE sequence (English encoder)
  - Example: "demain" attending to "tomorrow", "and", "creeps" (English)
  - Attention matrix: (target_len × source_len) - often rectangular!

This is how the decoder knows WHICH English words to focus on when
generating each French word during translation!
    """)

    print("\n" + "="*80)
    print("ARCHITECTURE IN TRANSFORMER TRANSLATION")
    print("="*80)
    print("""
Encoder (English):
  └─ Self-Attention: English words attend to each other
      Q, K, V all from English

Decoder (French):
  ├─ Self-Attention: French words attend to each other
  │   Q, K, V all from French (with masking for autoregressive generation)
  │
  └─ Cross-Attention: French words attend to English words ⭐
      Q from French, K and V from English ← THE KEY DIFFERENCE!
    """)


if __name__ == "__main__":
    main()
