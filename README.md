# Hand_angles_detection
### Related papers
https://drive.google.com/drive/folders/1xxPDvrQ9xO2RgIvJIwXZe_n1rn1aZpIZ?usp=sharing

### Usage
1. 下載整個程式檔（畫面中右上角有個綠色的 code, 點他之後下面會有個 download zip），下載完之後放到你想放的位置解壓縮。
2. 移動位址。右鍵點擊下載檔案並查看他所在的位置，打開 command line(終端機)，輸入以下指令，他會將目前的位置移到下載完的專案裡面。比如你把檔案下載到 C：\Users\USER\Desktop，且你下載的檔案名稱為 hand_angles_detection，那就要在終端機輸入 cd C：\Users\USER\Desktop\hand_angles_detection
```bash=
cd location_of_the_downloaded_dir
```
3. 將你想要分析的檔案放到專案檔案裡一個名叫 data 的資料夾，如果你想分析的是影片，就將檔案放到 data/videos; 如果想分析的是圖片，則將檔案放到 data/images 裡面。
4. 執行你想操作的程式，分成三種類型：單純做攝像頭測試、分析影片、分析圖片檔，選擇其中一種即可。這邊以分析影片為例，將 # analyze video 下面的程式碼複製貼到你的 command line 上
5. 根據需求調整超參數。
    * python 換成你系統底下偏好的執行檔（e.g., C:/Desktop/msys64/bin/python），否則他會使用系統預設的 python 執行檔
    * --filenames 後面的 xxx.mp4 換成你想分析的影片名稱，比如你想分析的影片是 abc.mp4 以及 bcd.mp4，則這邊就會改成 --filenames abc.mp4 bcd.mp4。另外，這邊的影片名稱要跟你放在 data/videos 裡面的完成完全一樣
    * --output_dir 後面的名稱指的是會將分析好的檔案存在哪邊，可以不用改他
    * --data_type 根據你想分析哪種形式的資料而定，只能輸入 videos 或 images
    * --start_frames 指的是你想從影片的哪個位置開始分析，單位是 frame，每一部影片會對應一個 start frame
    * --end_frames 指的是你想從影片的哪個位置結束分析，單位是 frame，每一部影片會對應一個 end frame, -1 指的是不中斷，直接分析到影片結束為止

```python=
# camera test
python camera_test.py

# analyze video
python detect_angles.py --filenames hands_1.mp4 hands_2.mp4 hands_3.mp4 hands_4.mp4 --output_dir Outputs --data_type videos --start_frames 0 0 0 0 --end_frames -1 -1 -1 -1

# analyze image
python detect_angles.py --filenames image_1.jpg image_2.jpg image_3.jpg --output_dir Outputs --data_type images
```
6. 查看結果。結果會存在 Outputs 資料夾裡面，裡面的存存形式為 output_dataname，比如你分析的是檔案名稱是 abc.mp4，則他的輸出結果就會在 output_abc，每個輸出結果都包含以下資料
    * detected images: 標示出手部位置的圖片檔案，單位是 frame，檔名為他在影片中的第幾個 frame
    * parameters.json: 包含一些影片相關資料以及使用的超參數，像是 fps(frame per second), start/end frame, 分析的成功率等等
    * left/right_hands_corrdinates.csv: 左/右手每個關節點位 (21 points) 在每個 frame 的 3D 位置，如果沒有檢測道則會在 Detected 這個欄位顯示 False
    * left/right_hands_angles.csv: 左/右手關節之間的角度，column 欄位分別對應每隻手指算出來的三種角度（_0, _1, _2），其中 _0 表示的是手腕到該手指第一個關節的角度，其他依此往外推。row 表示的是不同的 frame 所對應的角度
    * left/right_hands_angles_max_min.csv: 左/右手關節之間的角度的最大值、最小值、最大值產生的時間、最小值產生的時間，每個欄位可能會有多個數值，原因是因為病人可能會做多次的握拳動作，這邊會用自動檢測的方式輸出不同 round 的最大最小值。

### TODO
1. output the max angle and the min angle (each cycle)
2. detect the gradient of angles 

### 11/4 Meeting
* 測量方式：單次測量會有誤差，可以用多次測量再取平均值
* 之後做成 app 的時候，可以設計成標準的測量方式（像是 iphone 在做臉部辨識時的前置動作，讓我們把臉對準某一個輪廓 -> 靜置測量一陣子 -> 轉向測量 等等）
* mediapipe 不足點：大拇指到手腕之間的辨識不太行、關節測量點位是在中間而不是皮膚上下（與真實值會有誤差）

### 11/18 Progress
* Correct the computation of each angle 
* Compute the max angle and the min angle (each cycle)
    * Preprocess: smoothing (mean of rolling window)
    * Function: detect_max_min_by_threshold 
    * Problems
        * How to select an appropriate hyperparameter for the rolling window?
        * Detecting the maximization and the minization of angles by threshold may not work due to different threshold.
* Detect the gradient of angles 
    * There is no need for this. What we want is to display a specific delay of certain fingers. 

### TODO 
* Others
    * 兩根手指夾腳
    * 虎口角度
    * 體積計算（e.g., 大拇指轉一圈）
    * 3D camera 建議
* Outputs
    * (done) 小數點後一位（round(1)） 
    * (done) excel 表格不要有括號、不同 cycle 可以把他們展開
    * (done) 圖片不用存太多 
    * (done) video pixel status (e.g, 640*320)
    * (done) RIF(right index finger) -> 簡寫
    * 手部角度變化的圖形做衍伸（15 張圖)
        * （done) smooth 版本
        *  Title, name, ...
        *  NAN 處理 -> (若不處理，在 pandas 做 rolling mean 的時候會爆掉)
  
* Algorithms
    * cycle 定位有問題 -> 改進
    * 角度需要有 negative (-5 - 90) (角度不能夠對稱，相反過來應該要變負數 -> 但是內積沒有方向性) -> 使用 sin and cos 同時判斷
    * 影片辨識度改進（試一下不同的超參數，看一下能否不讓他做出不好的預測）
    * smooth 方式調研
    * mediapipe hand 做法調查（看 paper）
* Publications
    * application papers 寫法
* Evaluation
    * 臨床測試，並與套件結果做比較（e.g., ROM > 9成）


