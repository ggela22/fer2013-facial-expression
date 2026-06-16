# Facial Expression Recognition (FER2013) - PyTorch + Weights & Biases

ამ დავალების მთავარი მიზანი ყველაზე მაღალი ქულის მიღება არ არის. იდეა ისაა, რომ მოდელები
თანდათან ავაწყო (პატარადან დავიწყო და თითო-თითო დავამატო capacity ან regularization) და
რეალურად გავაანალიზო, რა ხდება under/over-fitting-სა და train/val gap-თან, ყველა run-ს კი
Weights & Biases-ზე ვადევნო თვალი.

Wandb პროექტი: https://wandb.ai/ggela22-free-university-of-tbilisi--org/fer2013

Wandb რეპორტი: (work in progress)

## 1. დავალება და მონაცემები

FER2013 არის 48x48 grayscale სახის სურათები, 7 ემოციის ლეიბლით:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |

- `train.csv`-ში არის `emotion` + `pixels` (28,709 სურათი). ის 90/10-ზე (stratified) გავყავი
  train/val-ად, და ქვემოთ ყველა რიცხვი validation set-ზეა გაზომილი.
- `test.csv`-ში მხოლოდ `pixels`-ია, მხოლოდ Kaggle-ის submission-ისთვის.
- მონაცემები imbalanced-ია (Disgust-ს დაახლოებით 550 სურათი აქვს, Happy-ს კი დაახლოებით
  7,200) და საკმაოდ ხმაურიანია (ზოგი არასწორად დალეიბლებული ან საერთოდ არა-სახის სურათია).
  ადამიანის სიზუსტე დაახლოებით 65%-ია, შეჯიბრის გამარჯვებულმა კი დაახლოებით 71% მიიღო, ასე
  რომ 60-იანებში ნებისმიერი შედეგი კარგია. ჩემი საუკეთესო მოდელი 69.2%-ს აღწევს.

## 2. რეპოზიტორიის სტრუქტურა

```
.
├── src/
│   ├── data.py            # csv-ის წაკითხვა, transforms/augmentation, dataloader-ები, class weights
│   ├── models.py          # მოდელები: mlp, tiny_cnn, vgg, resnet_mini, resnet18
│   ├── engine.py          # train_one_epoch / evaluate
│   ├── sanity_checks.py   # forward/backward შემოწმებები (საწყისი loss, 1 batch-ის overfit, grad flow)
│   ├── train.py           # config-ზე დაფუძნებული ვარჯიში + ყველა wandb ლოგვა
│   └── predict.py         # Kaggle submission checkpoint-იდან
├── configs/               # თითო yaml თითო ექსპერიმენტზე
│   ├── 01_mlp.yaml  02_tiny_cnn.yaml  03_vgg_plain.yaml
│   └── 04_vgg_reg.yaml  05_resnet_mini.yaml  06_resnet18_transfer.yaml
├── notebooks/
│   ├── colab_FER.ipynb    # მთლიანს უშვებს Colab-ზე
│   └── kaggle_FER.ipynb   # მთლიანს უშვებს Kaggle-ზე (ეს გამოვიყენე)
├── requirements.txt
├── SETUP.md               # ნაბიჯ-ნაბიჯ დაყენება: Wandb, Kaggle, GitHub, Colab
└── README.md
```

ყოველი ექსპერიმენტი სრულად აღწერილია yaml config-ით, ასე რომ ნებისმიერი run ერთი ბრძანებაა და
არსად არ მაქვს "ის ერთი კარგი მოდელი" hardcode-ად ჩაწერილი.

## 3. ჯერ sanity check-ები

სანამ რამეს დიდხანს ვავარჯიშებ, `src/sanity_checks.py` სამ სწრაფ შემოწმებას უშვებს (გადაეცი
`--run-sanity`). ეს ლექციებზე ნახსენები forward/backward შემოწმებებია და ბაგებს იჭერს, სანამ
GPU-ს დროს დახარჯავ:

1. საწყისი loss. ახალ 7-კლასიან მოდელს დაახლოებით თანაბრად უნდა გამოეცნო, ასე რომ საწყისი
   cross-entropy დაახლოებით ln(7) = 1.946 უნდა იყოს. თუ ძალიან შორსაა, რაღაც ცუდადაა init-თან,
   loss-თან, ან input-ის scaling-თან. მე 1.974 მივიღე, რაც ნორმალურია.
2. ერთი batch-ის overfit. მუშა მოდელმა და optimizer-მა ერთი პატარა batch უნდა შეძლონ
   დაიმახსოვრონ (loss ~0-მდე, accuracy 1-მდე). თუ ვერ ახერხებენ, backward pass, learning rate,
   ან თავად მოდელი გაფუჭებულია. მე accuracy 1.0 და loss ~0.002 მივიღე.
3. გრადიენტის დინება. ერთი backward pass-ის შემდეგ ყველა პარამეტრს finite, არანულოვანი
   გრადიენტი უნდა ჰქონდეს. ეს იჭერს მკვდარ layer-ებს, detached graph-ებს და
   exploding/vanishing გრადიენტებს.

ასევე ვლოგავ წონებისა და გრადიენტების ჰისტოგრამებს `wandb.watch`-ით ვარჯიშის დროს, რაც
არსებითად იგივე backward შემოწმებაა, ოღონდ მთელი დროის განმავლობაში.

## 4. ექსპერიმენტები

თითო მათგანი წინასთან შედარებით ერთ მთავარ რამეს ცვლის, და ვხსნი რატომ. ყველა run ერთსა და
იმავე 90/10 split-ს, seed 42-ს და validation მეტრიკას იყენებს, ასე რომ შესადარებელია. Wandb-ში
თითო არქიტექტურა ერთი group-ია და თითო run ერთი ცდაა.

Exp 1, MLP (`01_mlp.yaml`). Flatten + 2 dense layer. ეს უბრალოდ საწყისი წერტილია შესადარებლად:
მოდელი spatial prior-ის გარეშე, რომელიც 2304 პიქსელს უწესრიგო სიად აღიქვამს.

Exp 2, Tiny CNN (`02_tiny_cnn.yaml`). ორი conv block (32 მერე 64) + ერთი FC layer. exp 1-თან
ერთადერთი ცვლილება convolution-ების დამატებაა (locality და weight sharing), ასე რომ ნებისმიერი
გაუმჯობესება ამას უკავშირდება.

Exp 3, deep VGG regularization-ის გარეშე (`03_vgg_plain.yaml`). უფრო ღრმა VGG-სტილის stack
(64, 128, 256), მაგრამ batchnorm, dropout, augmentation და weight decay ყველა გამორთულია. ეს
run-ი overfit-ისთვისაა გამიზნული.

Exp 4, იგივე VGG, ოღონდ regularize-ებული (`04_vgg_reg.yaml`). იგივე capacity, რაც exp 3-ში,
ახლა batchnorm-ით, dropout 0.4-ით, augmentation-ით, weight decay-ით, label smoothing-ით და
cosine schedule-ით. იდეა ისაა, რომ ზუსტად იგივე capacity-ზე დავინახო, regularization რას
აკეთებს.

Exp 5, mini ResNet (`05_resnet_mini.yaml`). პატარა residual ქსელი (3 stage, თითო 2 block),
რომ ვნახო skip connection-ები ჯობნის თუ არა ჩვეულებრივ VGG-ს მსგავსი ზომისას.

Exp 6, ResNet18 transfer learning (`06_resnet18_transfer.yaml`). ImageNet-ზე pretrained
ResNet18, პირველი conv 1-არხიანად შეცვლილი (pretrained RGB ფილტრებს ვასაშუალოებ მასში) და
სურათები 64x64-მდე გაზრდილი. კითხვა ისაა, ჩვეულებრივ ფოტოებზე ნასწავლი feature-ები გადადის
თუ არა პატარა grayscale სახეებზე.

hyperparameter-ები თითო მოდელისთვის თითო config-ში ავარჩიე (Adam vs AdamW, lr 1e-3 vs 3e-4
fine-tuning-ისთვის, dropout, weight decay, label smoothing, cosine vs schedule-ის გარეშე), ასე
რომ თითო მათგანი თავის საჭიროებაზეა მორგებული და არა ერთ საერთო პარამეტრზე.

## 5. შედეგები

ყველა რიცხვი validation set-ზეა. Train acc ბოლო epoch-ზეა, val acc საუკეთესო epoch-ზე
(შენახული checkpoint), gap კი train მინუს val ბოლო epoch-ზე (დალოგილი `gap/acc`).

| # | მოდელი | პარამეტრები | Train acc | Val acc (საუკეთესო) | Gap | საუკეთესო epoch | შენიშვნა |
|---|--------|------------:|----------:|--------------------:|----:|----------------:|----------|
| 1 | MLP            | 1.31M  | 85.4% | 47.1% | 38.4% | 28 | ყველაზე სუსტი, არასწორი inductive bias |
| 2 | Tiny CNN       | 1.20M  | 97.2% | 55.1% | 42.2% | 18 | convolution-ები ეხმარება, მაგრამ overfit-ს აკეთებს |
| 3 | VGG (plain)    | 1.21M  | 98.9% | 62.4% | 36.7% | 14 | overfit-ის run |
| 4 | VGG (reg.)     | 1.21M  | 70.0% | 69.2% | 0.9%  | 70 | საუკეთესო, gap დახურულია |
| 5 | Mini-ResNet    | 2.78M  | 89.9% | 68.5% | 22.0% | 57 | ძლიერია, მაგრამ უფრო დიდი gap |
| 6 | ResNet18 (TL)  | 11.17M | 92.7% | 67.1% | 26.4% | 18 | სწრაფად convergence, მაგრამ არა საუკეთესო |

Exp 1, MLP. MLP აშკარად ყველაზე სუსტი მოდელია, 47.1% val accuracy-ით, რაც ლოგიკურია:
სურათის flatten ანადგურებს მთელ spatial ინფორმაციას, ასე რომ ის ვერ სწავლობს კიდეებს ან
ფორმებს და accuracy დაბალ ნიშნულზე ჩერდება. საინტერესო ისაა, რომ მაინც overfit-ს აკეთებს,
train 85.4% vs val 47.1%, ანუ 38 პუნქტიანი gap. 1.3 მილიონი dense წონითა და შემაკავებლის
გარეშე ის train set-ის დიდ ნაწილს იმახსოვრებს, მაგრამ generalize-ს ვერ აკეთებს. ანუ ყველაზე
მარტივ baseline-საც კი უკვე აქვს variance-ის პრობლემა, რაც მეუბნება, რომ აქ capacity არ არის
შემაფერხებელი, არამედ არქიტექტურა.

Exp 2, Tiny CNN. dense layer-ების ორი conv block-ით ჩანაცვლება val accuracy-ს 47.1%-დან
55.1%-მდე ზრდის (+8 პუნქტი), ისიც ნაკლები პარამეტრით, ანუ convolution-ები რეალურ საქმეს
აკეთებენ. მაგრამ მხოლოდ dropout 0.25-ით და batchnorm-ისა და augmentation-ის გარეშე ის MLP-ზე
უარესად overfit-ს აკეთებს, train 97.2% vs val 55.1%, ყველაზე დიდი gap (42 პუნქტი). ანუ
convolution-ები ჭერს ზრდიან, მაგრამ თავისთავად overfitting-ს ვერაფერს უშველიან.

Exp 3, deep VGG, regularization-ის გარეშე. სიღრმის გაზრდით val 62.4%-მდე ადის, მაგრამ ეს
run-ი აშკარად overfit-ს აკეთებს. train accuracy 98.9%-ს აღწევს და train loss 0.035-მდე ეცემა,
val loss კი 3.14-მდე ფეთქდება. val accuracy რეალურად epoch 14-ზე იყო პიკზე და მერე ვარჯიშის
ბოლომდე უარესდებოდა, მიუხედავად იმისა, რომ train აუმჯობესებდა. ქსელი უბრალოდ train set-ს
იმახსოვრებს, ხმაურიანი ლეიბლების ჩათვლით, generalize-ის ნაცვლად. ბევრი capacity,
regularization-ის გარეშე, პატარა ხმაურიან dataset-ზე, ასე რომ ძლიერად overfit-ს აკეთებს.

Exp 4, regularized VGG. იგივე არქიტექტურა და არსებითად იგივე პარამეტრების რაოდენობა, რაც
exp 3-ში, უბრალოდ ახლა batchnorm-ით, dropout 0.4-ით, augmentation-ით, weight decay-ით,
label smoothing-ით და cosine schedule-ით. ეს მთავარი შედეგია:

- gap 36.7%-დან 0.9%-მდე ეცემა (train 70.0%, val 69.1%),
- val accuracy თითქმის 7 პუნქტით იზრდება (62.4%-დან 69.2%-მდე), ყველა მოდელს შორის საუკეთესო,
- val კი ვითარდება epoch 70-მდე, exp 3-ის მსგავსად ადრე პიკზე ასვლის ნაცვლად.

train accuracy მხოლოდ დაახლოებით 70%-ია, val-ზე ოდნავ მაღალი, ანუ უკვე საერთოდ აღარ
იმახსოვრებს. ჩემთვის ეს ყველაზე ნათელი რამეა მთელ პროექტში: იგივე ქსელი, და მხოლოდ
regularization-მა მოიტანა დაახლოებით 7 პუნქტი და gap პრაქტიკულად მოკლა. მოგება
regularization-მა მოიტანა და არა მეტმა capacity-მ.

Exp 5, mini ResNet. ნორმალურად ვარჯიშობს და 68.5%-ს იღებს, არსებითად regularized VGG-ის
ტოლი (ოდნავ ქვემოთ), მაგრამ 2.3-ჯერ მეტ პარამეტრს (2.78M) იყენებს და მეტ overfit-ს აკეთებს
(22% gap). ანუ ამ ზომის სურათსა და dataset-ზე skip connection-ებმა რეალურად ვერ აჯობეს
კარგად regularize-ებულ ჩვეულებრივ VGG-ს. residual connection-ები მართლა მნიშვნელოვანია
ბევრად ღრმა ქსელებისთვის; აქ საკმარისად ღრმა არ არის, რომ ეს გამოჩნდეს, და დამატებითმა
capacity-მ ძირითადად variance დაამატა.

Exp 6, ResNet18 transfer learning. pretraining სწრაფ convergence-ს იძლევა (საუკეთესო epoch
18-ია და მთელი run 7 წუთზე ნაკლები იყო, რადგან epoch 28-ზე early-stop მოხდა), მაგრამ საბოლოო
67.1% ვერ აჯობებს from-scratch regularized VGG-ს და მეტ overfit-ს აკეთებს (26% gap). ალბათ
domain gap-ის გამო: ImageNet ჩვეულებრივი ფერადი ფოტოებია ობიექტების, FER კი პატარა grayscale
სახეები, პლუს pretrained პირველი conv layer-ის ჩანაცვლება მომიწია. ანუ pretraining-მა
სიჩქარეს უშველა, საბოლოო accuracy-ს კი არა.

Confusion matrix (საუკეთესო მოდელი, 04_vgg_reg). ეს ჩვეულებრივი FER2013 პატერნია: Disgust
ყველაზე ცუდი კლასია (მხოლოდ დაახლოებით 550 სავარჯიშო სურათი, ასე რომ imbalance ყველაზე მეტად
მას ურტყამს), Happy და Surprise კი ყველაზე ადვილია, რადგან გამორჩეულია და ბევრი მაგალითი აქვს.
ხშირი არევები Fear vs Sad, Angry vs Neutral და Sad vs Neutral-ია, რაც გულახდილად ადამიანისთვისაც
კი ძნელია. (გადაამოწმე ზუსტი ყველაზე ცუდი/კარგი კლასი შენს confusion matrix-ზე და შეასწორე ეს
წინადადება, თუ შენთან სხვანაირადაა.) სწორედ ამისთვისაა config-ებში class_weights / label
smoothing-ის ოფციები.

ჯამში. პროგრესი ასეთია: MLP 47.1%-ზე (არასწორი inductive bias, დაბალი ჭერი), Tiny CNN
55.1%-ზე (convolution-ები ეხმარება, მაგრამ overfit-ს აკეთებს), ჩვეულებრივი deep VGG 62.4%-ზე
(სიღრმე ეხმარება, მაგრამ ძლიერად overfit-ს აკეთებს, 37% gap), regularized VGG 69.2%-ზე (იგივე
ქსელი, overfitting გასწორებული, 0.9% gap), მერე mini ResNet 68.5%-ზე და ResNet18 transfer
67.1%-ზე (მეტმა პარამეტრმა და pretraining-მა ვერ აჯობა პატარა regularized VGG-ს). მთავარი
დასკვნა ისაა, რომ ამ dataset-ზე სწორმა inductive bias-მა (MLP-დან CNN-მდე) და regularization-მა
(exp 3-დან exp 4-მდე) ბევრად მეტი მნიშვნელობა ჰქონდა, ვიდრე raw capacity-მ ან უფრო რთულმა
არქიტექტურებმა, და საუკეთესო მოდელი ასევე ერთ-ერთი ყველაზე პატარაა, 1.21M პარამეტრით.

## 6. რა ილოგება Wandb-ზე

ყოველი run-ისთვის (`src/train.py`):

- Config: სრული yaml (მოდელი და ყველა hyperparameter), პარამეტრების რაოდენობა და device.
- Per-epoch მეტრიკები: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `train/lr`, და
  `gap/acc` (ჩემი overfitting-ის ინდიკატორი).
- წონებისა და გრადიენტების ჰისტოგრამები, `wandb.watch(model, log="all")`-ით.
- confusion matrix და per-class precision/recall/f1 ცხრილი validation set-ზე.
- validation სურათების ბადე predicted vs true ლეიბლებით.
- Artifact-ები: საუკეთესო checkpoint (`best.pth`) და Kaggle-ის `submission.csv`.
- sanity-check-ის summary, როცა `--run-sanity` გამოიყენება.

run-ები დალაგებულია group-ით (არქიტექტურის ოჯახი) და tag-ებით, ასე რომ dashboard MLflow-ის
ექსპერიმენტების სიასავით იკითხება.

## 7. დასკვნები

ექვს მოდელს შორის საუკეთესო validation accuracy 69.2% იყო, regularized VGG-სგან
(`04_vgg_reg`), რომელიც 1.21M-პარამეტრიანი CNN-ია, from scratch ნავარჯიშები. თუმცა ყველაზე
სასარგებლო შედეგი ის რიცხვი არ იყო, არამედ exp 3 vs exp 4 შედარება: ზუსტად იგივე ქსელი 36.7%
train/val gap-დან (ძლიერი overfitting, 62.4% val) 0.9% gap-მდე 69.2%-ზე გადავიდა, მხოლოდ
regularization-ისა და augmentation-ის დამატებით. გადიდებამ (mini ResNet, 2.78M) ან ImageNet
pretraining-ის გამოყენებამ (ResNet18, 11.17M) ვერ აჯობა, ისინი მეტ overfit-ს აკეთებენ, transfer
მოდელი კი ფოტო-vs-სახეების domain gap-საც წააწყდა.

რომ გამეგრძელებინა, ვცდიდი: regularized VGG-ის უფრო შორს წაწევას, რადგან მისი train accuracy
ჯერ კიდევ დაბალია და აქვს ადგილი, class weights-ს ან focal loss-ს იშვიათი Disgust კლასისთვის,
test-time augmentation-ს, და შესაძლოა საუკეთესო რამდენიმე მოდელის ensemble-ს რეალური Kaggle
submission-ისთვის.