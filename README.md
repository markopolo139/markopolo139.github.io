# Building Models like Software: Scaffolded Pretraining with Mixture of Training (MoT)

Large language model (LLM) pretraining has historically been an all-or-nothing, monolithic endeavor. Every layer in a massive Transformer is optimized jointly in a single, tightly coupled run. If a node fails, a loss spike occurs, or an architectural change is desired, engineers are forced to either roll back to a checkpoint or restart the entire system. 

The paper *Mixture of Training: Recombining Small-Scale Scaffolded Pretraining Runs into a Larger Language Model* introduces a divide-and-conquer alternative. The authors ask a fundamental question: can pretraining be decomposed into smaller, independently trainable depth slices that can later be recomposed into a fully functional language model?

## The Problem with Monolithic Training

In standard pretraining, a target Transformer $F$ composed of $K$ contiguous layer blocks is trained as a unified chain:

$$F = f_K \circ f_{K-1} \circ \dots \circ f_1$$

Gradients must flow end-to-end through all $K$ blocks simultaneously. This creates a scaling bottleneck, as all layers are locked together during optimization. 

## How Mixture of Training (MoT) Works

MoT decouples this pipeline by training each target block $f_i$ independently inside a frozen, pretrained **aligner scaffold**. 

### The Aligner Scaffold & Interface Contract
The primary challenge in training depth slices independently is hidden-state drift: if block $f_1$ and block $f_2$ train without communicating, the activation space output by $f_1$ will not match what $f_2$ expects as input. 

MoT resolves this by introducing a small, pretrained model called an **aligner** $A = a_K \circ \dots \circ a_1$, which is sliced into the same number of blocks $K$ as the target model. The aligner must be shape-compatible with the target model—sharing the global width, attention-head dimensions, feed-forward width, token embedding space, and output head. This allows layer blocks to be swapped cleanly without requiring additional projection layers.

### The Three-Stage Pretraining Lifecycle
MoT structures model construction into three distinct stages, allowing the model to be built piece by piece. 

**Stage 0: Aligner Preparation**
First, MoT selects or pretrains a shape-compatible aligner, then partitions both the aligner and the target architecture into $K$ contiguous layer blocks.

![Step 1: Using Pretrained Model as Aligner](/assets/images/mot-step1.png)
*Figure 1a: Slicing the pretrained aligner.*

**Stage 1: Parallel Scaffold Training**
To train a specific target block $f_i$, MoT constructs a dedicated scaffold network $S_i$:

$$S_i = a_K \circ \dots \circ a_{i+1} \circ f_i \circ a_{i-1} \circ \dots \circ a_1$$

Each target block is trained concurrently inside its own scaffold. 

![Step 2: Configuring the Target Larger Model](/assets/images/mot-step2.png)
*Figure 1b: Packaging the untrained target blocks.*

During optimization, only the parameters in the target block $f_i$ are updated using standard next-token prediction loss, while all aligner slices remain frozen. 

![Step 3: Replacing Aligner Components](/assets/images/mot-step3.png)
*Figure 1c: Scaffolded training. Because every target block optimizes inside the representational context of the exact same frozen aligner, they implicitly conform to a shared representational contract without exchanging gradients.*

**Stage 2: Recomposition & Adaptation**
Finally, the system discards the aligner slices and assembles the newly trained target blocks into the complete target model $\hat{F} = f_K \circ \dots \circ f_1$. 

![Step 4: Composing the Trained Submodels](/assets/images/mot-step4.png)
*Figure 1d: The "cold-composed" model is assembled. Optionally, a brief end-to-end adaptation pass reconciles boundary transitions.*

## Advantages and Disadvantages

By altering the engineering parameters of LLM pretraining, MoT introduces a new set of trade-offs:

**Advantages:**
* **Parallel Critical Path:** Because Stage 1 submodel jobs execute independently without inter-job gradient communication, training time scales with available parallel hardware, shortening the idealized dependency path compared to monolithic runs.
* **Modular Fault Tolerance:** Submodel runs are decoupled. A hardware crash or instability in one block's training job does not affect the others, simplifying recovery workflows.
* **Amortized Efficiency:** While pretraining an aligner requires compute upfront, the effective cost falls below a monolithic baseline if the same aligner is reused across multiple independent target-model runs.

**Disadvantages:**
* **Cold-Composition Mismatch:** Stitching submodels together directly introduces a performance mismatch at layer boundaries, requiring a Stage 2 adaptation pass to recover full baseline quality.
* **Single-Run Compute Overhead:** If the aligner's training compute is fully charged to a single target run without any reuse, the quality-parity schedule costs more total compute than a monolithic baseline.
* **Contiguous Granularity Constraints:** The method currently relies on contiguous depth slicing; non-contiguous or expert-level splits introduce permutation alignment challenges not handled by this scaffold setup.

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
