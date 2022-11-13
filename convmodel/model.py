from convmodel.tokenizer import ConversationTokenizer
from typing import List, Optional
import torch
import tqdm
import numpy as np
import random
import transformers
import math


def _show_progress_bar(seq, show: bool):
    return tqdm.tqdm(seq) if show else seq


def _set_reproducibility(seed: int = None, deterministic: bool = False):
    """
    Refer to the document for details
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)


class ConversationModel:
    def __init__(
        self,
        hf_tokenizer: ConversationTokenizer,
        hf_model: transformers.GPT2LMHeadModel,
        dataset_class,
        device: Optional[str] = None
    ):
        # Convert transformers Tokenizer to ConversationTokenizer
        tokenizer = ConversationTokenizer(tokenizer=hf_tokenizer)

        self._tokenizer = tokenizer
        self._hf_model = hf_model
        self._dataset_class = dataset_class

        # Set device
        if device:
            device = torch.device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        # Move model to device
        hf_model.to(device)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: Optional[str] = None):
        raise NotImplementedError("from_pretrained method must be implemented in a subclass")

    def generate(self, context: List[str], min_new_tokens: Optional[int] = None, **kwargs):
        raise NotImplementedError("generate method must be implemented in a subclass")

    @property
    def hf_tokenizer(self):
        return self._tokenizer.hf_tokenizer

    @property
    def hf_model(self):
        return self._hf_model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._device

    def save_pretrained(self, save_directory):
        self._tokenizer.save_pretrained(save_directory)
        self._hf_model.save_pretrained(save_directory)

    def fit(
        self,
        train_iterator,
        valid_iterator,
        output_path: Optional[str] = None,
        save_best_model: bool = False,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": 1e-4},
        warmup_steps: int = 10000,
        use_amp: bool = False,
        epochs: int = 1,
        accumulation_steps: int = 1,
        show_progress_bar: bool=True,
        log_steps: int = 100,
        shuffle_buffer_size: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        seed: Optional[int] = None,
        deterministic: bool = False,
        max_len: Optional[int] = None,
    ):
        """fit method enables train this model based on given train_iterator.

        Args:
            train_iterator (Iterator[ConversationExample]): iterator to use in training.

        """
        # Parameter assertion
        if save_best_model:
            assert output_path, f"Set output_path when you set save_best_model to True"

        # Set Reproducibility
        _set_reproducibility(seed=seed, deterministic=deterministic)

        # Prepare model
        model = self._hf_model

        # Prepare optimizer and scheduler
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
        )

        # Prepare data
        train_dataloader = self._dataset_class(
            iterator=train_iterator,
            tokenizer=self._tokenizer,
            max_len=model.config.n_positions if max_len is None else max_len,
        ).build_data_loader(
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        valid_dataloader = self._dataset_class(
            iterator=valid_iterator,
            tokenizer=self._tokenizer,
            max_len=model.config.n_positions if max_len is None else max_len,
        ).build_data_loader(
            # Do NOT need to shuffle validation data
            shuffle_buffer_size=None,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        # Setup scaler for SMP
        # Please refer to https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # variables to use in log
        num_steps = 0

        # keep best model loss
        best_val_loss = float("infinity")

        for epoch in range(1, epochs+1):
            # [*1] 学習モード
            model.train()

            for train_batch_idx, batch in _show_progress_bar(enumerate(train_dataloader, start=1), show=show_progress_bar):
                # ロスの計算グラフを構築する
                # forward 関数は、検証時にも利用するため別の関数で後で定義する
                with torch.cuda.amp.autocast(enabled=use_amp):
                    batch = {key: val.to(device=self._device) for key, val in batch.items()}
                    loss = model(**batch).loss
                    loss = loss / accumulation_steps

                # 勾配を計算し、その結果をテンソルの.gradに保存する
                scaler.scale(loss).backward()

                if train_batch_idx % accumulation_steps == 0:
                    # 勾配に従ってオプティマイザに登録したパラメータ (required_grad=Trueのテンソル) を更新
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    # [*2] 勾配の初期化Go
                    optimizer.zero_grad()

                    num_steps += 1

                # エポックのロス計算は、勾配計算を行わないため計算グラフを構築する必要はない。
                # 計算グラフを構築しないために item を使ってテンソルの中身を取り出して計算している。
                # item を使わないと計算グラフをバッチのループ毎に作り続けそれを train_loss にキープし続けるため、
                # メモリを大量に消費してしまう

                # ログの出力
                if train_batch_idx % (accumulation_steps * log_steps) == 0:
                    batch_log = dict(
                        epoch=epoch,
                        batch=train_batch_idx,
                        step=num_steps,
                        train_loss=loss.item(),
                        lr=optimizer.param_groups[0]['lr'],
                    )
                    print(batch_log)

                    # [*1] 検証モード
                    model.eval()
                    # [*3] 推論モードでは勾配計算しないので計算グラフを作成する必要がない。
                    #      `torch.no_grad()` コンテキスト内のテンソルの計算では計算グラフは構築されない。
                    with torch.no_grad():
                        val_loss = 0
                        for val_batch_idx, batch in _show_progress_bar(enumerate(valid_dataloader, start=1), show=_show_progress_bar):
                            batch = {key: val.to(device=self._device) for key, val in batch.items()}
                            loss = model(**batch).loss
                            val_loss += loss.item()

                    # Decide whether model is saved or not
                    val_loss_per_batch = val_loss/val_batch_idx
                    save_model = False

                    if output_path:
                        save_model = True
                        if val_loss_per_batch < best_val_loss:
                            best_val_loss = val_loss_per_batch
                        else:
                            if save_best_model:
                                save_model = False

                    # Save model
                    if save_model:
                        self.save_pretrained(output_path)

                    epoch_log = dict(
                        epoch=epoch,
                        valid_loss=val_loss_per_batch,
                        valid_ppl=math.exp(val_loss_per_batch),
                        save_model=save_model,
                    )
                    print(epoch_log)

                    # 学習モードに戻す
                    model.train()

    def eval(
        self,
        eval_iterator,
        batch_size: int=1,
        num_workers: int=0,
        prefetch_factor: int=2,
        max_len: Optional[int] = None,
    ):
        model = self._hf_model
        model.eval()

        eval_dataloader = self._dataset_class(
            iterator=eval_iterator,
            tokenizer=self._tokenizer,
            max_len=model.config.n_positions if max_len is None else max_len,
        ).build_data_loader(
            shuffle_buffer_size=None,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

        with torch.no_grad():
            eval_loss = 0
            for batch_idx, batch in _show_progress_bar(enumerate(eval_dataloader, start=1), show=_show_progress_bar):
                batch = {key: val.to(device=self._device) for key, val in batch.items()}
                loss = model(**batch).loss
                eval_loss += loss.item()

        eval_loss = eval_loss / batch_idx
        log = dict(
            eval_loss=eval_loss,
            eval_ppl=math.exp(eval_loss),
        )
        print(log)
