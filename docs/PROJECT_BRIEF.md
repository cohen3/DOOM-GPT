# DOOM in Transformer Activations

## Project idea

Build a funny but technically legitimate demonstration where frames from
DOOM are represented inside the activation space of a transformer.

The visual claim should be:

    "DOOM rendered in transformer activations"

not literally:

    "DOOM running on transformer weights"

because transformer weights are fixed parameters while activations depend
on the input.

The eventual demo should display a real DOOM frame next to a visualization
constructed directly from a transformer activation tensor.

---

# Core experiment

Choose a relatively small open transformer that fits comfortably on a
consumer GPU.

Initial candidates:
- Phi-family model
- another small Hugging Face causal language model if integration is easier

Do not hard-code the architecture deeply into the project.

Suppose an internal hidden state is:

    A ∈ R^(batch × tokens × hidden_dimension)

For batch size 1, flatten a selected activation tensor:

    A_flat = flatten(A)

Then select enough activation values for a framebuffer:

    P = A_flat[:H*W]

and reshape:

    screen = reshape(P, H, W)

Initially visualize this as a grayscale heatmap.

For a DOOM-like framebuffer, a useful eventual target is approximately
320 × 200, although development should start at lower resolutions such as
64 × 40 or 80 × 50 to make optimization easier.

---

# Phase 1: Activation viewer

Goal:

Run an input through a frozen transformer and visualize selected hidden
states.

Requirements:

- Load model and tokenizer.
- Freeze all model parameters.
- Allow model/layer selection.
- Capture hidden states.
- Display activation tensor dimensions.
- Flatten a deterministic subset.
- Reshape into a 2D activation framebuffer.
- Normalize only for visualization.
- Save the image.

Example conceptual pipeline:

    text
      ↓
    tokenizer
      ↓
    frozen transformer
      ↓
    layer L hidden state
      ↓
    flatten/select
      ↓
    reshape
      ↓
    heatmap

This phase does NOT need DOOM yet.

---

# Phase 2: Activation inversion / DOOM frame fitting

This is the first important experiment.

Take a target image F, eventually a DOOM frame.

The transformer remains frozen.

Instead of optimizing model weights, optimize a continuous input embedding
/ soft prompt Z.

Conceptually:

    Z
      ↓
    frozen transformer
      ↓
    selected hidden state A_L(Z)
      ↓
    projection / selection / reshape
      ↓
    predicted image F_hat

Optimize:

    loss(F_hat, F)

Initially use a simple loss such as mean squared error.

Only Z should receive gradients.

Transformer parameters must remain unchanged.

Pseudo-objective:

    minimize_Z || framebuffer(A_L(Z)) - F ||²

Important:

The framebuffer must be derived directly from transformer activation
values.

Do not train a decoder network in this phase because that would make it
much easier for the decoder to hide the actual image representation.

Start at low image resolution.

Useful experiments:

- different transformer layers;
- different soft-prompt lengths;
- different activation subsets;
- raw activations vs normalized activations;
- MSE vs other image losses;
- optimization convergence curves.

Save:
- target image;
- initial activation image;
- reconstructed image;
- loss curve;
- experiment metadata.

---

# Phase 3: Real DOOM frames

Introduce ViZDoom or another legal/easy-to-use DOOM environment.

Capture frames programmatically.

Preprocess frames to the experiment resolution.

Run Phase 2 against actual game frames.

Build a visualization containing:

    Original DOOM frame | Transformer activation framebuffer

with metadata:

    model
    layer
    tensor shape
    soft prompt length
    optimization step
    loss

---

# Phase 4: Sequence

Optimize successive frames.

Start by independently fitting frames.

Then investigate initialization of frame t+1 using the optimized soft prompt
from frame t.

Question:

Does temporal similarity between consecutive DOOM frames make activation
inversion faster?

This can become an interesting research experiment in addition to the joke.

---

# Phase 5: Learned encoder

Instead of optimizing a soft prompt from scratch for every frame, train a
small encoder:

    DOOM frame
        ↓
    encoder E
        ↓
    soft prompt Z
        ↓
    frozen transformer
        ↓
    activations
        ↓
    framebuffer

The transformer stays frozen.

This phase investigates whether arbitrary images can efficiently be mapped
into a controllable region of transformer activation space.

---

# Stretch goal: Neural DOOM

Much later, investigate whether game state/history/actions can produce the
next visual state.

Conceptually:

    previous state + actions
             ↓
        learned system
             ↓
        transformer
             ↓
    activation framebuffer
             ↓
       next DOOM frame

This is significantly more difficult and should NOT be attempted until the
activation inversion experiment works reliably.

---

# Scientific questions

Besides creating a funny demo, investigate:

1. Which transformer layers are easiest to control?

2. How much soft-prompt capacity is required to encode an image?

3. Does reconstruction quality scale with prompt length?

4. Are some activation subspaces significantly easier to manipulate?

5. How stable is the representation under activation ablation?

6. Can a prompt optimized for layer L also produce recognizable structure
   in neighboring layers?

7. Can consecutive video frames reuse nearby regions of input embedding
   space?

8. How much information is really encoded versus introduced by visualization
   normalization?

---

# Important anti-cheating rules

The demo must not:

- render the original DOOM image on top of the heatmap;
- use the target frame during visualization except for side-by-side comparison;
- modify transformer weights during the activation inversion experiment;
- introduce a powerful image decoder during Phase 2;
- call arbitrary projection output "transformer activations" without clearly
  documenting the transformation.

Any normalization or reshaping must be documented.

---

# First success criterion

The first meaningful success is NOT running a complete DOOM game.

It is:

Given one static target image, optimize only an input soft prompt so that a
2D matrix constructed directly from a frozen transformer's internal
activations visibly reconstructs the target.

Everything else comes afterward.