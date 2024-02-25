# Finetuning RAG embeddings

<p align="center">
<img src = "https://github.com/SwamiKannan/Finetuning-RAG-Embeddings/blob/main/images/cover.png"><br>
<sub> Credit: <a href="https://www.freepik.com/premium-photo/tuning-radio-radio-station_16035333.htm">Freepik</a></sub>
</p>

Fine tuning embeddings is super important when the distribution of the data you want to process is different from the typical distribution of data on which the original embedding model is trained on. Basically, if the data in your vectorstore is science and research-related but the data that was used for training the embeddings is business related, then similar words may have different connotations e.g. "relative" in physics may refer to Einstein's theory of relativity but in business, it may be more relevant to content on family businesses. Hence, for specialized topics, it makes sense to further fine tune the embeddings to re-orient the relation of words to each other.
Hence, Fine-tuning is the process of adjusting your embedding model to better fit the domain of your data. Though before fine-tuning it yourself, you should always take a look at the Hugging Face model database and check if someone already fine-tuned a model on data that is similar to yours.
