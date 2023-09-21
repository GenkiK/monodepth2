1. $\boxed{\large \quad A \quad }$ の実現は重要.
    * $\boxed{\large \quad A \quad }$: Training monocular **absolute** depth estimation model without LiDAR sensor
    
2. これまでの研究は $\boxed{\large \quad A \quad }$ に対して $\boxed{\large \quad B \quad }$ を行ってきた. しかしそれらの手法では $\boxed{\large \quad A \quad }$ を真に解いた（実現した）とは言えない．なぜならば...だから．すなわち $\boxed{\large \quad A \quad }$ のポイントは
$\boxed{\large \quad A' \quad }$ である.
    * $\boxed{\large \quad B \quad }$: use velocity, IMU/GPS, or camera height as supervision to learn metric scale
        <!-- * In other words, their models learn metric scale relying on GT labels unrelated to visual cues. -->
    * ...: They need another measurement besides RGB camera sensor. Thus, their method cannot leverage a large amount of video data in the wild (e.g. Youtube) for training, though [Kick back & relax: learning to reconstruct the world by watching SlowTV](https://arxiv.org/pdf/2307.10713.pdf)(ICCV'23) shows that training MDE model with more data makes it more accurate.
    * $\boxed{\large \quad A' \quad }$: learning metric information with only scale priors independent of specific environments
    <!-- * $\boxed{\large \quad A' \quad }$: learning metric information with only scale priors independent of specific environments (and predictable from visual cues) -->
        * **Motivated by the observation that humans perceive a spatial scale of a visual input mainly on prior knowledge of object sizes, we use object size priors to learn metric information.**

1. $\boxed{\large \quad A' \quad }$ は自明な問題ではない．なぜならば $\boxed{\large \quad C_1 \quad }$, $\boxed{\large \quad C_2 \quad }$, $\dots$ であるから．
    * $\boxed{\large \quad C_1 \quad }$: When there are no objects in a frame, there are no scale constraints on training.
    * $\boxed{\large \quad C_2 \quad }$: To leverage object size priors for metric scale learning, the model need to know some parts defining object dimensions, e.g., 3D BBox. However, it is not easy especially when objects can be seen partially because they are occluded or cut off from the image.
    * $\boxed{\large \quad C_3 \quad }$: Though most of the previous works on other tasks such as scale recovery for monocular visual odometry and single view metrology try to model an object size prior as a probability distribution, finding its parameter is problematic.
        * For instance, [Scale-aware Insertion of Virtual Objects in Monocular Videos](https://arxiv.org/pdf/2012.02371.pdf)(ISMAR'20) builds GMMs as object size priors by scraping object size data. しかし，特に車などの人工物については，ただ販売されている車のデータを収集するだけでなく流通量なども考慮しなければ，現実世界のサイズ分布との間に乖離が生じてしまう．実際，異なるデータセット間で車のサイズの分布を比較すると，その差は無視できるほど小さくない．
        * 母集団分布とモデル化した分布のズレがそのまま学習するスケールのズレにつながる
    <!-- * $\boxed{\large \quad C_2 \quad }$: On an image plane, finding line segments defining object dimensions is not easy. Besides, some objects in an image might be seen only partially because they are occluded or cut off from the image. -->

2. 我々は $\boxed{\large \quad C_1 \quad }$, $\boxed{\large \quad C_2 \quad }$, $\dots$ を $\boxed{\large \quad E \quad }$ というアイデアに基づいた $\boxed{\large \quad D_1 \quad }$, $\boxed{\large \quad D_2 \quad }$, $\dots$ によって解決する．
    $\boxed{\large \quad E \quad }$: object height priorから得たスケール情報をカメラ高さに集約し，カメラ高さの一定性を仮定することでこれを全シーンに渡って最適化する
    $\boxed{\large \quad E \quad }$: **Aggregating scale information from object height priors into a camera height and optimizing it through whole scenes under the constraint that it is constant**

3. 各 $\boxed{\large \quad C_i \quad }$ について，我々は $\boxed{\large \quad D_i \quad }$ を行う．
    * $\boxed{\large \quad D_1 \quad }$: 以下のtraining schemeにより，シーン全体で一貫したメトリックスケールを学習できる．また物体が登場しないフレームにおいてもスケール制約を与えられる．
        1. 推定したDepth mapの道路領域からカメラ高さを計算
        2. Object height priorを元に各フレームのスケールファクターを計算（$D_2$）し，1.で計算したカメラ高さをスケール
        3. 全フレームで計算したスケール後のカメラ高さを最適化（mean or median）．これを教師としてカメラ高さをsupervise
            * 道路環境において，道路領域が全く見えていない状況はほとんどない
            * 自動運転の文脈において，カメラ高さの一定性を仮定するのは自然

    * $\boxed{\large \quad D_{2a} \quad }$: 物体シルエットの高さとObject Height Priorを比較して各フレームのスケールファクターを決定
        * 道路平面上に垂直に立てた平面に，推定したDepth mapから得られる物体領域のpoint cloudを正射影してシルエットを取得
        * 物体の見え方（姿勢・空間的な位置）に関係なくObject size(height) priorを利用できる

    * $\boxed{\large \quad D_{2b} \quad }$: horizonと物体の画像上の位置・大きさから導かれるおおまかな物体高さとobject height priorとの乖離から「異常な物体」を特定し，スケールファクターの決定時に除外する
        * 「異常な物体」: 物体上部の見切れ/道路平面に接地してない/セグメント領域誤り/クラス分類誤りを含む物体
        1. Depth mapの道路領域からhorizonを計算
        2. [Putting Objects in Perspective](https://www.ri.cmu.edu/pub_files/pub4/hoiem_derek_2006_1/hoiem_derek_2006_1.pdf)(IJCV'08)の近似を適用すると，以下図の関係式が成立する．この式を元に計算した3Dの物体高さとobject height priorの差を計算し，閾値よりも差が大きければ異常物体とみなす．

    * $\boxed{\large \quad D_3 \quad }$: 簡単にスクレイピング可能な車画像と車種が対になったデータを元に，見た目の情報から高さを推定するモデルを学習させる

    * その他のアドバンテージ
        * 任意のself-supervised MDE modelに適用可能なので，長年積み重ねられてきた知見がそのまま活かせる
        * 直接メトリックスケールのDepth mapを出力するので，推論時のコストは増えない

    * その他小さい工夫
        * self-supervised MDE modelでは，特に遠くの物体についてlossの性質上形状が不正確になり，道路平面に対して平行に近い形状を推定しがち．これを一定の範囲に収め，スケールファクター決定時の精度を上げるために，Object height priorと2D BBoxのピクセル高さから導かれる粗い幾何的制約を導入する．また，この制約（＝損失）に対して掛ける係数を学習が進むに連れて小さくすることで，制約を加えることにより生じる不正確性を取り除く．
            * 粗い制約により，物体深度を鉛直な平面として推定するようになるが，学習が進むに連れて徐々に正しい形状になっていく
            * 反対に最適化したカメラ高さの損失には，徐々に大きくなる係数をかけている．これは学習の序盤ではカメラ高さの値が信頼できないから．

    * 簡単に追加可能な工夫（$\boxed{\large \quad A' \quad }$ の解決策というよりかはより精度を上げるための工夫）
        * シルエットの高さを計算する際はdetachしてからmax/min関数を適用している．これをsoftmax/softmin関数で代用して微分可能な形にした値とobject height priorの差を損失とすることで，物体深度に$$対してスケールの制約が加わり，より正確になる(かも)

        * この手法は道路領域深度をうまく推定できないと話にならない．Monodepth2の推定結果を見る限り，道路平面に対するカメラロールは過小に推定されがち．これを解決するために，ロールのaugmentationを加える
            * 探した感じロールのaugmentationしてる論文はなかった．
            * ただaugmentationするだけじゃなくて，ロールした後の画像で推定したDepthとロールする前の画像で推定したDepthが一致するように制約を加えても良さそう

self-supervised MDEモデルは scale factor ambiguity を持っている．だからobject size priorをマルチフレームで活用するのは容易ではない
https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Can_Scale-Consistent_Monocular_Depth_Be_Learned_in_a_Self-Supervised_Scale-Invariant_ICCV_2021_paper.pdf
おそらく
→scale-invariant: 定数倍するだけで現実と一致する（Midasにはない特徴）
→scale-consistent: スケールが一貫している
**"scale factor ambiguity"って言葉なにかに使えない？**

**窓領域が無限に飛ぶのに対してロバスト**