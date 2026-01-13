import tiktoken
import torch

from cfg_gpt2 import GPT_CONFIG_124M
from gpt2 import GPTModel
from load_data import create_dataloader_v1
from trainer import train_model_simple


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def clean_text(text: str) -> str:
        lines = text.splitlines()
        cleaned = []

        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(line)

        return '\n'.join(cleaned)

    with open(r'data/gutenberg/sherlock_holmes.txt', encoding='utf8') as f:
        raw_text = f.read()
        cleaned_text = clean_text(raw_text)

    train_ratio = 0.9
    split_idx = int(train_ratio * len(cleaned_text))
    cleaned_text_train = cleaned_text[:split_idx]
    cleaned_text_val = cleaned_text[split_idx:]

    train_dataloader = create_dataloader_v1(
        txt=cleaned_text_train,
        batch_size=8,
        drop_last=True,
        max_length=GPT_CONFIG_124M['context_length'],
        num_workers=0,
        shuffle=True,
        stride=GPT_CONFIG_124M['context_length'] // 2,
    )
    val_dataloader = create_dataloader_v1(
        txt=cleaned_text_val,
        batch_size=8,
        drop_last=True,
        max_length=GPT_CONFIG_124M['context_length'],
        num_workers=0,
        shuffle=True,
        stride=GPT_CONFIG_124M['context_length'] // 2,
    )

    tokenizer = tiktoken.get_encoding('gpt2')
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=0.01,
        lr=3e-4,
        betas=(0.9, 0.95),
    )

    train_model_simple(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        model=model,
        optimizer=optimizer,
        device=device,
        tokenizer=tokenizer,
        eval_freq=5,
        eval_iter=4,
        num_epochs=10,
        start_context='My name is ',
    )


if __name__ == '__main__':
    main()
