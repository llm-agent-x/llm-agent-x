import re
from typing import List, Any
from dataclasses import dataclass
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from .LLMMerger import MergeOptions
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK data (if you haven't already)
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")


# --- Configuration ---
# Higher means sentences must be more similar to be considered for insertion.
INSERTION_SIMILARITY_THRESHOLD = 0.6
# Higher means sentences must be very similar to be flagged as redundant.
REDUNDANCY_SIMILARITY_THRESHOLD = 0.9
# Model for sentence embeddings
MODEL_NAME = "all-MiniLM-L6-v2"  # A good general-purpose model


class AlgorithmicMerger:
    def __init__(self, options: MergeOptions):
        self.options = options  # Store the MergeOptions (including LLM, though we won't use it directly)
        self.model = SentenceTransformer(
            MODEL_NAME
        )  # Load sentence transformer model here

    def _preprocess_text(self, text):
        # Normalize whitespace and handle paragraph breaks better
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize multiple newlines to double
        text = re.sub(
            r"(?<!\n)\n(?!\n)", " ", text
        )  # Single newlines (not paragraph breaks) to space
        return text.strip()

    def _segment_into_sentences(self, text):
        # Preserve paragraph structure by splitting by double newlines first
        paragraphs = text.split("\n\n")
        all_sentences = []
        para_markers = []  # Keep track of paragraph endings

        for para_idx, para in enumerate(paragraphs):
            if not para.strip():
                if (
                    para_idx > 0
                    and para_markers
                    and para_markers[-1] == len(all_sentences) - 1
                ):
                    # Avoid double paragraph markers if previous sentence ended a para
                    pass
                elif (
                    all_sentences
                ):  # Only add a paragraph marker if there are preceding sentences
                    para_markers.append(
                        len(all_sentences) - 1
                    )  # Mark end of previous para
                continue

            sentences = nltk.sent_tokenize(para.strip())
            if sentences:
                all_sentences.extend(sentences)
                para_markers.append(
                    len(all_sentences) - 1
                )  # Mark the last sentence of this paragraph

        # Remove last marker if it's for the very last sentence (implicit paragraph end)
        if para_markers and para_markers[-1] == len(all_sentences) - 1:
            para_markers.pop()

        return all_sentences, para_markers

    def _get_embeddings(self, sentences):
        if not sentences:
            return np.array([])
        return self.model.encode(sentences, convert_to_tensor=False)

    def merge_documents(self, documents: List[str]) -> str:
        if not documents:
            return ""

        # Assuming we want to merge all documents into the first one, one-by-one
        if len(documents) == 1:
            return documents[0]

        merged_text = documents[0]
        for i in range(1, len(documents)):
            merged_text = self._merge_two_texts(merged_text, documents[i])

        return merged_text  # Return the final merged text

    def _merge_two_texts(
        self,
        text1_raw,
        text2_raw,
        insertion_threshold=INSERTION_SIMILARITY_THRESHOLD,
        redundancy_threshold=REDUNDANCY_SIMILARITY_THRESHOLD,
    ):

        text1_processed = self._preprocess_text(text1_raw)
        text2_processed = self._preprocess_text(text2_raw)

        sents1, para_markers1 = self._segment_into_sentences(text1_processed)
        sents2, _ = self._segment_into_sentences(
            text2_processed
        )  # Para markers for text2 less critical for this strategy

        if not sents1 and not sents2:
            return ""
        if not sents1:
            return text2_processed  # Use processed to keep paragraph structure
        if not sents2:
            return text1_processed  # Use processed

        embeds1 = self._get_embeddings(sents1)
        embeds2 = self._get_embeddings(sents2)

        # --- Match sentences from text2 to text1 ---
        # For each sentence in text2, find its best match in text1
        # Store as: {index_in_sents1: [ (original_index_in_sents2, sentence2_text, similarity_score), ... ]}
        insertion_map = {i: [] for i in range(len(sents1))}
        unmatched_sents2 = (
            []
        )  # Sentences from text2 that don't meet insertion_threshold

        # Keep track of text2 sentences already assigned to prevent duplicates in merged output
        used_sents2_indices = set()

        if len(sents2) > 0 and len(sents1) > 0:  # Ensure embeds2 is not empty
            similarity_matrix = cosine_similarity(
                embeds2, embeds1
            )  # rows: sents2, cols: sents1

            for i2, sent2 in enumerate(sents2):
                if not embeds1.size:  # No sentences in text1 to compare against
                    unmatched_sents2.append(sent2)
                    continue

                best_match_s1_idx = np.argmax(similarity_matrix[i2])
                best_match_score = similarity_matrix[i2][best_match_s1_idx]

                # Check for redundancy with the best match in text1
                # This is a simple check. A more advanced one would check against *all* sents1.
                s1_match_text = sents1[best_match_s1_idx]
                # Using a simple length-normalized LCS as a quick redundancy check alongside embedding
                # This helps catch near-identical phrasing that embeddings might still score < 1.0
                if best_match_score >= redundancy_threshold:
                    # A more sophisticated check could be added here, e.g., edit distance
                    # For now, high cosine similarity is the main driver for redundancy
                    used_sents2_indices.add(i2)  # Mark as used to avoid appending later
                    continue

                if best_match_score >= insertion_threshold:
                    insertion_map[best_match_s1_idx].append(
                        (i2, sent2, best_match_score)
                    )
                else:
                    unmatched_sents2.append(sent2)

        # --- Construct the merged text ---
        merged_sentences = []
        current_merged_embeddings = []  # For more advanced dynamic redundancy check

        for i1, sent1 in enumerate(sents1):
            merged_sentences.append(sent1)

            # Get text2 sentences anchored to this text1 sentence
            # Sort them by their original order in text2 to maintain local coherence
            # and then by similarity (desc) as a tie-breaker if needed.
            # Using original_index primarily ensures flow.
            sents2_to_insert = sorted(
                insertion_map.get(i1, []), key=lambda x: x[0]
            )  # x[0] is original_index_in_sents2

            for original_i2, sent2_text, sim_score in sents2_to_insert:
                if original_i2 not in used_sents2_indices:
                    merged_sentences.append(sent2_text)
                    used_sents2_indices.add(original_i2)

            # Add paragraph break if s1 was the end of a paragraph
            if (
                i1 in para_markers1 and i1 < len(sents1) - 1
            ):  # Don't add for the very last sentence
                if merged_sentences and not merged_sentences[-1].endswith(
                    "\n\n"
                ):  # Avoid double breaks
                    merged_sentences.append("\n\n")

        # Append remaining unmatched sentences from text2 that weren't used
        # (and weren't deemed redundant initially)
        if unmatched_sents2:
            # Add a paragraph break if the main text didn't end with one
            if merged_sentences and not merged_sentences[-1].endswith("\n\n"):
                merged_sentences.append("\n\n")
            for sent2 in unmatched_sents2:
                # Find the original index of sent2 to check if it was already used via insertion_map
                # This is a bit inefficient here; ideally, `unmatched_sents2` should only contain truly unused ones.
                # For simplicity, we'll just check if its text is already present in the tail of merged_sentences
                # to avoid direct repetition if it was somehow added.
                # A better way is to filter unmatched_sents2 based on used_sents2_indices *before* this loop.

                # Re-filter unmatched_sents2 to be sure
                original_indices_of_unmatched = []
                for i_s2, s_s2_text in enumerate(sents2):
                    if (
                        s_s2_text in unmatched_sents2
                        and i_s2 not in used_sents2_indices
                    ):
                        original_indices_of_unmatched.append(s_s2_text)

                if (
                    original_indices_of_unmatched
                ):  # If there are truly unmatched and unused sentences
                    if merged_sentences and not merged_sentences[-1].endswith("\n\n"):
                        merged_sentences.append(
                            "\n\n"
                        )  # Separator for unmatched content

                    for s_text in original_indices_of_unmatched:
                        merged_sentences.append(s_text)

                    # Break after processing the filtered list once
                    break

        # Join sentences, handling the paragraph markers
        final_text_parts = []
        for part in merged_sentences:
            if part == "\n\n":
                if final_text_parts and not final_text_parts[-1].endswith("\n\n"):
                    # Remove trailing space before double newline for cleaner output
                    if final_text_parts[-1].endswith(" "):
                        final_text_parts[-1] = final_text_parts[-1][:-1]
                    final_text_parts.append("\n\n")
                elif not final_text_parts:  # If it's the first element, don't add
                    pass
            else:
                final_text_parts.append(part)

        # Final join, ensuring spaces between sentences but not before/after explicit paragraph breaks
        output_text = ""
        for i, part in enumerate(final_text_parts):
            if part == "\n\n":
                output_text = output_text.rstrip(
                    " "
                )  # Remove space before paragraph break
                output_text += "\n\n"
            else:
                output_text += part
                if i < len(final_text_parts) - 1 and final_text_parts[i + 1] != "\n\n":
                    output_text += " "  # Add space if next part isn't a paragraph break

        return output_text.replace(
            " \n\n", "\n\n"
        ).strip()  # Clean up spaces before newlines
