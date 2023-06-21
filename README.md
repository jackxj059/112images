# EmbeddedImageProcessingFinal

## Environment
### Python Interpreter Version == 3.8.10
- 下載專案

  `git clone https://github.com/jackxj059/112images.git`

- 進入112images資料夾

  `cd 112images`

- 安裝所需套件

  `pip install -r requirements.txt`

## Usage
### For Morphology
- 切換分支

  `git checkout master`

- 進入src資料夾

  `cd src`

- 執行程式

  `python main.py <video path>` => `python main.py ../data/1.mp4`

### For Harris Corner Detection:
- 切換分支

    `git checkout harris`
- 進入src資料夾

  `cd src`

- 執行程式

  `python main.py <video path>` => `python main.py ../data/1.mp4`

程式執行後，會跳出image視窗，左鍵點擊並拖曳，框選屬於是馬路的地方，再按下空白鍵

視窗說明: 
- frame是當前的原始畫面
- fg_mask是opencv背景模型抓出來的前景
- draw是程式執行最後的結果
- 按下q後可以將畫面全部關閉
