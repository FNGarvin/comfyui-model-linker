"""
Fuzzy Matcher Module

Implements fuzzy string matching to find similar model names.
"""

import os
import posixpath
import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz as _rapidfuzz
except ImportError:  # optional - difflib fallback keeps the matcher dependency-free
    _rapidfuzz = None


def normalize_path_separators(path: str) -> str:
    """
    Convert backslashes to forward slashes.

    Workflows are shared across OSes, so a path authored on Windows
    (subdir\\model.safetensors) must compare equal to the same path on
    Linux (subdir/model.safetensors). os.path functions can't do this:
    on Linux a backslash is a regular filename character.
    """
    return path.replace('\\', '/') if path else ''


def normalize_path(path: str) -> str:
    """Normalize a path for cross-OS comparison (separators + redundant parts)."""
    if not path:
        return ''
    return posixpath.normpath(normalize_path_separators(path))


def extract_filename(path: str) -> str:
    """Get the final path component regardless of separator style."""
    return normalize_path_separators(path).rstrip('/').rsplit('/', 1)[-1]


def normalize_filename(filename: str) -> str:
    """
    Normalize a filename for comparison.
    
    Removes file extension, converts to lowercase, and normalizes
    separators (underscores, hyphens, spaces).
    
    Args:
        filename: Filename to normalize
        
    Returns:
        Normalized string for comparison
    """
    # Remove file extension
    base = os.path.splitext(filename)[0]
    
    # Convert to lowercase
    base = base.lower()
    
    # Normalize separators: replace underscores, hyphens, dots, and spaces with a single space
    base = re.sub(r'[_\-.\s]+', ' ', base)

    # Strip whitespace
    base = base.strip()

    return base


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity score between two strings (0.0 to 1.0).

    Uses rapidfuzz when installed, otherwise SequenceMatcher.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score from 0.0 (completely different) to 1.0 (identical)
    """
    if _rapidfuzz is not None:
        return _rapidfuzz.ratio(str1, str2) / 100.0
    return SequenceMatcher(None, str1, str2).ratio()


# ==================== MODEL FAMILY SCORING ====================
#
# Shared workflows constantly reference a different precision or quant of
# the same model (flux1-dev-fp8 vs flux1-dev, Q4_K_M vs Q8_0). Those are
# near-certain substitutes, while a different parameter count or model
# generation (qwen3-4b vs qwen3-8b, qwen2 vs qwen3) is not - even though
# plain edit distance rates both kinds of pair about the same.

# Precision / quant tokens: fp8, bf16, int4, q4, iq3, tq1, nvfp4, e4m3fn, bnb4 ...
_VARIANT_TOKEN = re.compile(
    r'^(?:(?:bf|bfloat|fp|float|int|uint|f|i)\d+(?:mixed|scaled)?'
    r'|(?:i?q|tq)\d+|(?:mxfp|nvfp)\d+|e\d+m\d+(?:fnuz|fn)?|bnb\d+)$'
)
_VARIANT_WORDS = {
    'gguf', 'ggml', 'gptq', 'awq', 'exl2', 'hqq', 'aqlm', 'nf4', 'mixed', 'scaled',
    'pruned', 'ema', 'emaonly', 'quant', 'quantized', 'converted', 'compressed',
}
# Quant heads like q4 are followed by suffix tokens: q4 k m, q5 k s, q8 0, iq4 xs
_QUANT_HEAD = re.compile(r'^(?:i?q|tq)\d+$')
_QUANT_TAIL = {'0', '1', '4', '8', 'k', 'm', 's', 'l', 'xl', 'xs', 'xxs', 'nl'}
# Parameter count: 4b, 8b, 1.5b, 350m
_CAPACITY_TOKEN = re.compile(r'^\d+(?:\.\d+)?[bmk]$')

# Score floors/caps (0-1 scale)
SAME_FAMILY_SCORE = 0.94
SAME_SIGNATURE_SUBSET_SCORE = 0.90
SAME_SIGNATURE_SCORE = 0.84
CONFLICT_CAP = 0.69


def is_variant_token(token: str) -> bool:
    """True for precision/quantization/format tokens (fp8, q4, gguf, ...)."""
    return token in _VARIANT_WORDS or bool(_VARIANT_TOKEN.match(token))


def family_tokens(normalized: str) -> List[str]:
    """
    Tokens of a normalized filename with precision/quant tokens removed.

    'flux1 dev fp8 e4m3fn' -> ['flux1', 'dev']; 'model q4 k m' -> ['model']
    """
    normalized = re.sub(r'\bema only\b', 'emaonly', normalized)
    tokens = []
    in_quant_tail = False
    for token in normalized.split():
        if in_quant_tail and token in _QUANT_TAIL:
            continue
        in_quant_tail = bool(_QUANT_HEAD.match(token))
        if is_variant_token(token):
            continue
        tokens.append(token)
    return tokens


def model_signature(tokens: List[str]) -> Optional[Tuple[str, str]]:
    """
    (architecture, size) from family tokens, e.g. ['qwen3', '4b', 'it'] ->
    ('qwen3', '4b'). None when there's no parameter-count token or the part
    before it is too short to name an architecture.
    """
    for i, token in enumerate(tokens):
        if _CAPACITY_TOKEN.match(token):
            arch = ''.join(tokens[:i])
            if len(arch) >= 3 and re.search(r'[a-z]', arch):
                return arch, token
            return None
    return None


def numeric_conflict(arch1: str, arch2: str) -> bool:
    """Same name apart from version digits - qwen2 vs qwen3, sd15 vs sd21."""
    letters1, letters2 = re.sub(r'\d', '', arch1), re.sub(r'\d', '', arch2)
    digits1, digits2 = re.findall(r'\d+', arch1), re.findall(r'\d+', arch2)
    return letters1 == letters2 and digits1 != digits2


def calculate_filename_confidence(target_filename: str, candidate_filename: str) -> float:
    """
    Similarity (0.0-1.0) between two model filenames, aware of model families.

    - identical after normalization -> 1.0 (the only way to reach 1.0)
    - otherwise the better of the plain and word-order-independent ratio
    - same model, different precision/quant -> at least 0.94
    - same architecture+size, other differences -> at least 0.84/0.90
    - different size or generation of one architecture -> at most 0.69
    """
    target_norm = normalize_filename(extract_filename(target_filename))
    candidate_norm = normalize_filename(extract_filename(candidate_filename))
    if target_norm == candidate_norm:
        return 1.0

    best = max(
        calculate_similarity(target_norm, candidate_norm),
        calculate_similarity(' '.join(sorted(target_norm.split())),
                             ' '.join(sorted(candidate_norm.split()))),
    )

    target_family = family_tokens(target_norm)
    candidate_family = family_tokens(candidate_norm)
    if target_family and candidate_family:
        if target_family == candidate_family:
            best = max(best, SAME_FAMILY_SCORE)
        else:
            best = max(best, calculate_similarity(' '.join(target_family),
                                                  ' '.join(candidate_family)))
            target_sig = model_signature(target_family)
            candidate_sig = model_signature(candidate_family)
            if target_sig and candidate_sig:
                if target_sig == candidate_sig:
                    target_set, candidate_set = set(target_family), set(candidate_family)
                    subset = target_set <= candidate_set or candidate_set <= target_set
                    best = max(best, SAME_SIGNATURE_SUBSET_SCORE if subset else SAME_SIGNATURE_SCORE)
                elif target_sig[0] == candidate_sig[0]:
                    best = min(best, CONFLICT_CAP)  # different parameter count
                elif target_sig[1] == candidate_sig[1] and numeric_conflict(target_sig[0], candidate_sig[0]):
                    best = min(best, CONFLICT_CAP)  # different model generation

    return min(best, 0.999)


def calculate_similarity_with_normalization(str1: str, str2: str) -> float:
    """
    Calculate similarity score with filename normalization.
    
    Normalizes both strings before comparing.
    
    Args:
        str1: First string (typically model filename)
        str2: Second string (typically candidate model filename)
        
    Returns:
        Similarity score from 0.0 to 1.0
    """
    norm1 = normalize_filename(str1)
    norm2 = normalize_filename(str2)
    return calculate_similarity(norm1, norm2)


def find_matches(
    target_model: str,
    candidate_models: List[Dict[str, str]],
    threshold: float = 0.0,
    max_results: int = 10,
    expected_categories: List[str] = None
) -> List[Dict[str, any]]:
    """
    Find similar models using fuzzy matching.

    Args:
        target_model: The target model filename/path to match
        candidate_models: List of candidate model dictionaries with 'filename' or 'path' key
        threshold: Minimum similarity score (0.0 to 1.0) to include in results
        max_results: Maximum number of results to return
        expected_categories: Folder categories the referencing node loads from.
            Candidates outside these are flagged 'category_mismatch' — the
            node couldn't actually load them from where they are (issue #3)

    Returns:
        List of match dictionaries sorted by similarity (highest first,
        right-folder matches before wrong-folder ones at equal similarity):
        {
            'model': original model dict from candidates,
            'filename': model filename,
            'similarity': similarity score (0.0 to 1.0),
            'confidence': confidence percentage (0 to 100),
            'category_mismatch': True if the file is in a folder the node
                                 doesn't load from,
            'expected_categories': the expected categories (when known)
        }
    """
    matches = []

    def category_mismatch(candidate):
        if not expected_categories:
            return False
        candidate_category = candidate.get('category')
        if not candidate_category:
            return False  # Unknown candidate category - don't flag
        return candidate_category not in expected_categories
    
    # Normalize path separators to forward slashes so Windows- and
    # Linux-authored workflow paths compare identically on either OS
    target_model_normalized = normalize_path(target_model)

    # Extract just the filename from target_model (remove any subfolder paths)
    # target_model might be just a filename or might include subfolder paths
    target_filename = extract_filename(target_model)

    for candidate in candidate_models:
        # Get filename from candidate (prefer 'filename' key, fallback to extracting from 'path' or 'relative_path')
        candidate_filename = candidate.get('filename')
        candidate_path = candidate.get('path', '') or candidate.get('relative_path', '')
        
        # If no filename key, try to extract from path or relative_path
        if not candidate_filename:
            if candidate_path:
                candidate_filename = extract_filename(candidate_path)

        if not candidate_filename:
            continue

        # Normalize candidate paths for cross-OS comparison
        candidate_path_normalized = normalize_path(candidate_path)
        candidate_relative_path = candidate.get('relative_path', '')
        candidate_relative_path_normalized = normalize_path(candidate_relative_path)

        # Check if normalized paths are identical (100% match)
        # This handles cases where paths differ only by separator (e.g., path/to/model vs path\to\model)
        # The target is usually relative, so check both the absolute and relative candidate paths
        path_match = False
        if target_model_normalized:
            if candidate_path_normalized and candidate_path_normalized == target_model_normalized:
                path_match = True
            elif candidate_relative_path_normalized and candidate_relative_path_normalized == target_model_normalized:
                path_match = True
        
        if path_match:
            # Exact path match after normalization = 100% confidence
            similarity = 1.0
            matches.append({
                'model': candidate,
                'filename': candidate_filename,
                'similarity': similarity,
                'confidence': round(similarity * 100, 1),
                'category_mismatch': category_mismatch(candidate),
                'expected_categories': expected_categories
            })
            continue
        
        # Compare just filenames (not paths); only an exact normalized match
        # reaches 100%, everything else is capped below it
        similarity = calculate_filename_confidence(target_filename, candidate_filename)
        
        # Only include if above threshold
        if similarity >= threshold:
            matches.append({
                'model': candidate,
                'filename': candidate_filename,
                'similarity': similarity,
                'confidence': round(similarity * 100, 1),  # Convert to percentage
                'category_mismatch': category_mismatch(candidate),
                'expected_categories': expected_categories
            })

    # Sort by similarity (highest first); at equal similarity, right-folder
    # matches rank above wrong-folder ones
    matches.sort(key=lambda x: (x['similarity'], not x['category_mismatch']), reverse=True)
    
    # Limit to max_results
    matches = matches[:max_results]
    
    return matches

