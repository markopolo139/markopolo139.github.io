<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>

# Building Models like Software: Scaffolded Pretraining with Mixture of Training (MoT)

Large language model (LLM) pretraining has historically been an all-or-nothing, monolithic endeavor. Every layer in a massive Transformer is optimized jointly in a single, tightly coupled run. If a node fails, a loss spike occurs, or an architectural change is desired, engineers are forced to either roll back to a checkpoint or restart the entire system. 

The paper *Mixture of Training: Recombining Small-Scale Scaffolded Pretraining Runs into a Larger Language Model* introduces a divide-and-conquer alternative. The authors ask a fundamental question: can pretraining be decomposed into smaller, independently trainable depth slices that can later be recomposed into a fully functional language model?

## The Problem: Hidden-State Drift

In standard pretraining, a target Transformer $F$ composed of $K$ contiguous layer blocks is trained as a unified chain:

$$F = f_K \circ f_{K-1} \circ \dots \circ f_1$$

Gradients flow end-to-end through all $K$ blocks simultaneously. This creates a scaling bottleneck because all layers are locked together during optimization. 

Why can't we simply split the model, train the first half on GPU A, the second half on GPU B, and glue them together later? The answer is **hidden-state drift**. If block $f_1$ and block $f_2$ train without communicating, the mathematical activation space output by $f_1$ will not match the input format that $f_2$ has learned to expect. Stitching them together "cold" results in a catastrophic collapse in quality. 

## The Solution: The Aligner as an "Interface Contract"

MoT resolves this mismatch by introducing a small, pretrained model called an **aligner** ($A = a_K \circ \dots \circ a_1$). The aligner is sliced into the exact same number of blocks ($K$) as the target model. 

Crucially, the aligner acts as a universal translator or a standardized "API contract" for the neural network. It must be shape-compatible with the target model—sharing the exact same global width, attention-head dimensions, token embeddings, and output head. This shape compatibility allows layers to be swapped cleanly without requiring messy post-hoc projection layers. While the aligner shares the target model's hidden width ($d_{\text{model}}$), it is much shallower—meaning a single thin aligner block $a_i$ (e.g., 1–2 layers) acts as a stand-in for a much deeper target block $f_i$ (e.g., 6–8 layers).

To train a specific target block (e.g., $f_i$), MoT builds a "scaffold" network around it using the aligner:

$$S_i = a_K \circ \dots \circ a_{i+1} \circ f_i \circ a_{i-1} \circ \dots \circ a_1$$

During training, only the target block $f_i$ is updated; the surrounding aligner layers are strictly frozen. Because every independent target block optimizes inside the exact same frozen aligner environment, they all learn to speak the same mathematical language. They conform to a shared representational contract without ever exchanging gradients with each other.

### The Three-Stage Pretraining Lifecycle
MoT structures model construction into three distinct stages, allowing the model to be built piece by piece. 

![Mixture of Training (MoT) Architecture](/assets/images/mot_approach.png)
*Figure 1: The complete architecture and workflow of Mixture of Training (MoT). Source: Sabry et al. (2026) [1].*

Based on the architecture detailed in Figure 1, the MoT process unfolds as follows:

*   **Step 1: Using Pretrained Model as Aligner (Stage 0).** A shape-compatible pretrained aligner is selected and partitioned into contiguous layer blocks. Its parameters are strictly frozen.
*   **Step 2: Configuring the Target Larger Model for Training (Stage 0).** The target architecture's randomly initialized layers are packaged into submodels (e.g., early layers, middle layers, last layers) corresponding to the aligner's partitions.
*   **Step 3: Scaffolded Training (Stage 1).** For each partition $i$, the aligner block $a_i$ is swapped out for the deeper, randomly initialized target block $f_i$ to form the scaffold $S_i$. Each scaffold is trained independently on standard next-token prediction loss: activations pass through the frozen prefix ($a_1 \dots a_{i-1}$), get transformed by the active target block ($f_i$), and pass through the frozen suffix ($a_{i+1} \dots a_K$) to compute gradients exclusively for $f_i$.
*   **Step 4: Composing the Trained Submodels (Stage 2).** Once all scaffolds finish training, the frozen aligner slices are discarded, and the trained target blocks are stitched directly together into the full target model ($\hat{F} = f_K \circ \dots \circ f_1$). This "cold-composed" model then undergoes a brief end-to-end adaptation pass to smooth out any subtle activation mismatches at the block boundaries.

## Advantages and Disadvantages

By altering the engineering parameters of LLM pretraining, MoT introduces a new set of trade-offs:

**Advantages:**
* **Parallel Critical Path:** Because Stage 1 submodel jobs execute independently without inter-job gradient communication, training time scales with available parallel hardware, shortening the idealized dependency path compared to monolithic runs.
* **Modular Fault Tolerance:** Submodel runs are decoupled. A hardware crash or instability in one block's training job does not affect the others, simplifying recovery workflows.
* **Amortized Efficiency:** Pretraining the aligner introduces a one-time compute overhead. If a team only trains a single target model, MoT costs more total compute than a baseline run. However, if the exact same aligner is reused to scaffold three or more independent target models, the effective cost drops below a monolithic baseline.

**Disadvantages:**
* **Cold-Composition Mismatch:** Stitching submodels together directly introduces a performance mismatch at layer boundaries, requiring a Stage 2 adaptation pass to recover full baseline quality.
* **Single-Run Compute Overhead:** If the aligner's training compute is fully charged to a single target run without any reuse, the quality-parity schedule costs more total compute than a monolithic baseline.
* **Contiguous Depth Constraints:** The scaffold only enforces alignment at the input and output boundaries of sequential layer blocks. Extending MoT to non-sequential layers or width-wise splits (like training parallel MoE experts separately) fails because internal neurons exhibit permutation symmetry—meaning separately trained width slices will not align their internal feature dimensions without extra weight-matching algorithms.

## Experimental Takeaways

The authors evaluated MoT on a 1.3B-parameter language model trained on the C4 dataset, revealing several key mechanistic insights:

* **The Aligner is Essential:** Omitting the aligner scaffold altogether leads to a sharp degradation in model quality upon composition. The frozen aligner is strictly required to enforce boundary compatibility across the independently trained slices.
* **Adaptation Closes the Gap:** While "cold composition" leaves a performance gap compared to a monolithic run, a brief Stage 2 adaptation pass closes most of the difference, and a longer quality-parity schedule reaches the same reported perplexity as the baseline.
* **Data Diversity Helps:** Training different submodels on disjoint data streams during Stage 1 actually improves cold-composition quality. Because the shared aligner scaffold maintains structural interface alignment, the recomposed target model benefits from broader aggregate token exposure.
* **The Granularity Trade-off:** Increasing the number of splits (e.g., from 2 blocks to 4 blocks) reduces the compute required for Stage 1 and increases parallelism, but the increased number of block interfaces results in a higher cold-composition penalty. 

By proving that language model pretraining can be factorized into independently trainable components, MoT opens a concrete design space for reusable training units and decentralized model development.

---

### References
1. *Sabry, M., Augenstein, S., Rush, K., & Dery, L. (2026). Mixture of Training: Recombining Small-Scale Scaffolded Pretraining Runs into a Larger Language Model. arXiv:2608.13277v1 [cs.CL].*
