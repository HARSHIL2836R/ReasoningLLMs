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

