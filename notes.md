### April 7
Tried running `t5-small`,`facebook/opt-1.3b` and `openai-community/gpt2` but all gave wrong answers, 0% accuracy. Will have to go through code and check for more models.  

### April 11
optillm is an OpenAI API compatible _optimizing inference proxy_ which implements several state-of-the-art techniques that can improve the accuracy and performance of LLMs.  
Inference proxy is a software layer that sits between the LLM and the application, allowing for optimization of the LLM's performance and accuracy.  
It can be used to improve the performance of LLMs by optimizing the way they process and generate text.

### April 12
 We first prompt the language model
with chain-of-thought prompting, then instead of greedily decoding the optimal reasoning path, we
propose a “sample-and-marginalize” decoding procedure: we first sample from the language model’s
decoder to generate a diverse set of reasoning paths; each reasoning path might lead to a different
final answer, so we determine the optimal answer by marginalizing out the sampled reasoning paths
to find the most consistent answer in the final answer set.

## April 14
First, a language model is prompted with a set of manually written chain-of-thought exemplars (a few chain of thought demonstrations are provided asexemplars in prompting).  
Next, we sample a set of candidate outputs from the language model’s decoder, generating a diverse set of
candidate reasoning paths.  
Finally, we marginalize over the sampled reasoning paths to find the most consistent answer in the final answer set.  
Self-consistency explores an interesting space between open-ended text generation and optimal
text generation with a fixed answer.  
One should note that self-consistency can be applied
only to problems where the final answer is from a fixed answer set, but in principle this approach can
be extended to open-text generation problems if a good metric of consistency can be defined between
multiple generations, e.g., whether two answers agree or contradict each other.

---

### Decoder-Only Models

**Architecture**: These models consist solely of the decoder component from the original Transformer architecture.

**Functionality**: They are designed for autoregressive text generation, predicting the next token in a sequence based on preceding tokens.

**Examples**: GPT series (e.g., GPT-2, GPT-3, GPT-4), PaLM, LaMDA, Falcon.

**Use Cases**:
- Open-ended text generation.
- Conversational AI.
- Code generation.
- Creative writing.

**Advantages**:
- Efficient for tasks requiring text continuation or generation.
- Simpler architecture leads to faster inference times.

**Limitations**:
- Less effective for tasks requiring deep understanding of input context, such as translation or summarization.
- May struggle with tasks that involve complex input-output mappings.

---

### Encoder-Decoder Models

**Architecture**: These models incorporate both encoder and decoder components. The encoder processes the input sequence into a context-rich representation, which the decoder then uses to generate the output sequence.

**Examples**: T5, BART, mT5, MarianMT.

**Use Cases**:
- Machine translation.
- Text summarization.
- Question answering.
- Data-to-text generation.

**Advantages**:
- Effective for tasks requiring a strong alignment between input and output sequences.
- Better at handling tasks involving complex transformations of the input.

**Limitations**:
- More computationally intensive due to the dual components.
- Longer inference times compared to decoder-only models.

---

### Comparative Summary

| Aspect              | Decoder-Only Models                     | Encoder-Decoder Models                      |
|---------------------|-----------------------------------------|---------------------------------------------|
| **Architecture**    | Only decoder component                  | Both encoder and decoder components         |
| **Primary Use**     | Text generation                         | Sequence-to-sequence tasks                  |
| **Examples**        | GPT series, PaLM, LaMDA, Falcon         | T5, BART, mT5, MarianMT                     |
| **Strengths**       | Efficient text generation               | Effective input-output alignment            |
| **Limitations**     | Less context understanding              | Higher computational requirements           |

---

### Creative Insights

The choice between decoder-only and encoder-decoder architectures depends on the specific requirements of the task. Decoder-only models are well-suited for generative tasks where the model needs to produce coherent and contextually relevant text based on a prompt. In contrast, encoder-decoder models excel in tasks where understanding and transforming the input sequence into a different output sequence is essential.

Recent research has explored hybrid approaches and adaptations of these architectures to leverage the strengths of both. For instance, some studies have investigated using decoder-only models for sequence-to-sequence tasks by incorporating mechanisms to better handle input context, aiming to bridge the gap between the two architectures.

---

Sampling scheme. To sample diverse reasoning paths, we followed similar settings to those
suggested in Radford et al. (2019); Holtzman et al. (2020) for open-text generation. In particular, for
UL2-20B and LaMDA-137B we applied temperature sampling with T = 0.5 and truncated at the
top-k (k = 40) tokens with the highest probability, for PaLM-540B we applied T = 0.7, k = 40, and
for GPT-3 we use T = 0.7 without top-k truncation. We provide an ablation study in Section 3.5 to
show that self-consistency is generally robust to sampling strategies and parameters.
 
We evaluate self-consistency over four transformer-based lan-guage models with varying scales:
- UL2 (Tay et al., 2022) is an encoder-decoder model trained on a mixture of denoisers with 20-
billion parameters. UL2 is completely open-sourced4 and has similar or better performance than
GPT-3 on zero-shot SuperGLUE, with only 20B parameters and thus is more compute-friendly;
- GPT-3 (Brown et al., 2020) with 175-billion parameters. We use two public engines code-davinci-
001 and code-davinci-002 from the Codex series (Chen et al., 2021) to aid reproducibility;5
- LaMDA-137B (Thoppilan et al., 2022) is a dense left-to-right, decoder-only language model with
137-billion parameters, pre-trained on a mixture of web documents, dialog data and Wikipedia;
- PaLM-540B (Chowdhery et al., 2022) is a dense left-to-right, decoder-only language model with
540-billion parameters, pre-trained on a high quality corpus of 780 billion tokens with filtered
webpages, books, Wikipedia, news articles, source code, and social media conversations.

For some tasks (e.g., ANLI-R1, e-SNLI, RTE),
adding chain-of-thought does hurt performance compared to standard prompting (Brown et al., 2020),
but self-consistency is able to robustly boost the performance and outperform standard prompting,
making it a reliable way to add rationales in few-shot in-context learning for common NLP tasks.

Comparison to Sample-and-Rank: One commonly used approach to improve generation quality is
sample-and-rank, where multiple sequences are sampled from the decoder and then ranked according
to each sequence’s log probability (Adiwardana et al., 2020). We compare self-consistency with
sample-and-rank on GPT-3 code-davinci-001, by sampling the same number of sequences from the
decoder as self-consistency and taking the final answer from the top-ranked sequence.

Comparison to Beam Search: In Table 6, we compare self-consistency with beam search decoding
on the UL2-20B model. For a fair comparison we report the accuracy under the same number of
beams and reasoning paths.

One limitation of self-consistency is that it incurs more computation cost. In practice people can try a
small number of paths (e.g., 5 or 10) as a starting point to realize most of the gains while not incurring
too much cost, as in most cases the performance saturates quickly (Figure 2). As part of future work,
one could use self-consistency to generate better supervised data to fine-tune the model, such that the
model can give more accurate predictions in a single inference run after fine-tuning. In addition, we
observed that language models can sometimes generate incorrect or nonsensical reasoning paths (e.g.,
the StrategyQA example in Table 4, the two population numbers are not exactly correct), and further
work is needed to better ground models’ rationale generations.

