import os
from dataset import get_dataset, get_tokenizer
from model import TranscribeModel
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

torch.autograd.set_detect_anomaly(True)

vq_initial_loss_weight = 10
vq_warmup_steps = 1000
vq_final_loss_weight  = 0.5
num_epochs = 1000
starting_steps = 0
num_examples = None
model_id = "test37"
num_batch_repeats = 1

starting_steps = 0
BATCH_SIZE = 64
LEARNING_RATE = 0.005

def run_loss_function(log_probs, target, blank_token):
    loss_function = nn.CTCLoss(blank=blank_token)
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))
    target_lengths = (target != blank_token.sum(dim=1))
    target_lengths = tuple(t.item() for t in target_lengths)
    input_seq_first = log_probs.permute(1,0,2)
    loss = loss_function(input_seq_first, target, input_lengths)
    return loss



def main():
    log_dir = f"runs/speech2text_training/{model_id}"
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    tokenizer = get_tokenizer()
    blank_token = tokenizer.token_to_id("□")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pastModelPath = f"models/{model_id}/model_latest.pth"
    if os.path.exists(f"models/{model_id}/model_latest.pth"):
        print(f"loading model: {model_id}")
        model = TranscribeModel.load(pastModelPath)
    else:
        model = TranscribeModel(
            num_codebooks=2,
            codebook_size=32,
            embedding_dim=32,
            num_transformer_layers=2,
            vocab_size=len(tokenizer.get_vocab()),
            strides=[2,6,4],
            initial_mean_pooling_kernel_size=4,
            max_seq_length=400
        ).to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{num_trainable_params} Parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = get_dataset(
        batch_size = BATCH_SIZE,
        num_examples=num_examples,
        num_workers=1
    )

    ctc_losses = []
    vq_losses = []
    num_batches = len(dataloader)
    steps = starting_steps

    for i in range(num_epochs):
        for idx, batch in enumerate(dataloader):
            for repeat_batch in range(num_batch_repeats):
                #may have to update these based on dataset
                audio = batch["audio"]
                target = batch["input_ids"]
                target = batch["text"]

                if target.shape[1] > audio.shape[1]:
                    print(f"Padding Audio. audio shape is {audio.shape}, target shape is {target.shape}")
                    audio = torch.nn.functional.pad(audio, (0,0,0,target.shape[1]-audio.shape[1]))
                    print(f"Audio padded to shape: {audio.shape}")

                    audio = audio.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()
                    output, vq_loss = model(audio)
                    #compute loss
                    ctc_loss = run_loss_function(output, target, blank_token)

                    #interpolate loss weight if needed
                    vq_loss_weight = max(
                        vq_final_loss_weight, 
                        vq_initial_loss_weight - (vq_initial_loss_weight-vq_final_loss_weight)*(steps/vq_warmup_steps)
                    )
                    if vq_loss is None:
                        loss = ctc_loss
                    else:
                        loss = ctc_loss + vq_loss_weight * vq_loss
                    if torch.isinf(loss):
                        print("inf loss, skipping step")
                    #update gradients
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                    ctc_losses.append(ctc_loss.item())
                    vq_losses.append(vq_loss.item())
                    steps +=1

                    if steps%20 ==0:
                        avg_ctc_loss = sum(ctc_losses)/len(ctc_losses)
                        avg_vq_loss = sum(vq_losses)/len(vq_losses)
                        avg_loss = avg_ctc_loss + vq_loss_weight * avg_vq_loss
                        print(f"Average loss: {avg_loss}")

