# DotImporter

一応画像から頂点を自動で打つスクリプト

ライン画像から均等に頂点を打ったり DotImporterとは一体

# インストール前に

Blenderに画像処理系Pythonパッケージを追加してもらう必要がある

コマンドプロンプトを管理者として実行

```bash
cd C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin
```

```bash
python.exe -m pip install scikit-image pillow scipy numpy
```

これらを実行する

あとはGithubからReleasesのZipでもダウンロードすればいいんじゃないの？

# 制作例

このような素材があったとする


なんとかしてフォトショでラインのみ残るように加工

BlenderのDotImportタブで各パラメータを設定

読み込む画像を選択

ConvertモードをLineに

今回はドローンの飛行範囲はW 90m H 120mなので RangeXに90 RangeYに120

ドローンの数は500なので MaxPointsに500と入力

DetectCreateを押して生成すると

概ね良い感じに生成される

余りがあるのでとりあえず帽子の中を埋めようと思う

まずはこのように頂点を選択

StoreCustomを押して選択状態を保存

UseCustomRegionにチェックを入れる

余った頂点を選択して、ApplyPlacementを押す

なんとなく頂点が配置できた

# UI解説

Image 変換したい画像パス

OutputDir 後述のCSV書き出しで使用 設定しないとシーンが保存されているパスに書き出す

### Detection 画像処理関連

Convert 画像処理方法

- None 変換無し ドット画像から読み込みのときに使う
- Line ライン画像

    BlurRadius テクニカルパラメーター 基本触らなくてよい
    
    ThreshScale  テクニカルパラメーター 基本触らなくてよい
    
    JunctionRaito テクニカルパラメーター 基本触らなくてよい
    
- Shape 塗りつぶされた画像
    
    Fillモード
    
    Shapeから生成する場合はOutline内部を埋める処理をすることができます
    
    None 何もしない
    
    Grid グリッド状にシェイプ内部にも頂点を生成
    
    SemiGrid 上記の半分生成されるVer
    
    Topology ノーマル方向にアウトラインを内側に押し込んだ仮想ラインにドットを打つ
    
    つまりいい感じに流れのある点が出来る
    
    Random ランダムに点を打つ
    

**Invert** 背景が白の画像ではOffで良い

Threshold テクニカルパラメーター 基本触らなくてよい

ResizeMax 画像サイズ大きいとめちゃ重くなるの防止にリサイズするための最大サイズ指定

### Placement 生成関連

Unit per Pixel テクニカルパラメーター 基本触らなくてよい

RangeX ドローンの飛行区域の幅メートル

RangeY 同様に縦

VertexSpacing 頂点同士の間隔メートル

Origin 生成メッシュのセンター

FlipY 上下反転 なんであるのかは知らん

Collection 格納コレクション名

Object 生成メッシュ名

MaxPoint 頂点の上限 形状的に打ち切れなくても余った分は端に自動でグリッド状に配置される

ReserveBottom 検知した画像に使って良い頂点の割合 50%なら半分は必ず頂点が余る

AutoExpandLimit 必要に応じて頂点上限を上げる？ なんで、なんのためにあるのかは忘れた

### HeightMap(まだ有効な場面がなかったためWIP)

HeightMap 画像指定

AutoHeight そのとおり自動で起伏をつけてくれるとかなんとか

SaveCSV SkybrushUtilのCSVで読み込めるものを生成するかどうか 一応モードによっては色も入る

Detect Create 実行

### ManualPlacement 余った頂点を散らせる

Placement Fillモードと同じようなもん 対象の個数が面積に対して少ないなら間隔自動で変化

IgnoreExsitingVertices 選択頂点以外を参照して間隔維持するか

UseMeshRegion 平面メッシュから散らす領域を指定

UseCustomRegion 頂点から領域指定 頂点を領域の輪郭と捉える

StoreCustom 領域頂点指定

ApplyPlacement 動かす頂点を選んで実行 面積に対して選択分配置しきれなければ不要分は動かない 

### MeshToDots(動作未確認)

メッシュのエッジをサンプリングして配置するとかなんとか

### Path to Dots (動作未確認)

パスをサンプリングして配置するとかなんとか

# 仕様

コーナー 交差 末端を検知 そこを基準にそこから点を打ち始めます

# 不具合

ライン生成でのベース画像の推奨サイズが1kです 大きすぎると生成結果が望ましいものではなくなってしまいます
