# Nishika: Fake News detection

https://www.nishika.com/competitions/27/summary

## Approach

- `rinna/japanese-gpt2-medium` で分類器を学習
    - `nlp-waseda/roberta-base-japanese`, `cl-tohoku/bert-large-japanese`, `rinna/japanese-gpt-1b` に比べて手元の実験で良好な結果を示した
    - ただし `rinna/japanese-gpt-1b` はメモリ制約のため出力部から遠い層を固定したので、厳密な比較ではない
    - 生成に用いた事前学習済みモデルが必ずしも検知で有効とは限らない点は https://arxiv.org/abs/2011.01314 でも考察されている
- 学習データが3000件程度と少なかったため、最初は全量データで学習し、順位表のスコアを参考に評価データに擬似ラベルを付与して学習データに加えた
- 1行目の後に `[SEP]` を入れる
    - 言語モデルによる生成文でも、1文目など prompt 部分は人間が生成した文章であるため
- エラー分析での考察をもとに後処理
    - 「C」や「47」といった存在しない媒体の処理など

## Run

```bash
python fulldata_training.py
python pseudo_5fold_training.py
python submission.py
```

## Appendix

[村山ら](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/PH1-16.pdf)によると、情報科学の研究では「フェイクニュース」について狭義と広義の定義がある。
事実性だけを基準とする広義の定義では本コンペのような言語モデルによる生成文をフェイクニュースと見なす立場も理解できる一方で、多数派なのは発信者の意図などを重視する狭義の定義である点には留意が必要である。
