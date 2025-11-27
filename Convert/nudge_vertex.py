import bpy
import bmesh
from mathutils import Vector, kdtree


def relax_vertices_min_distance_kdtree(obj, min_dist=0.1, iterations=10):
    """
    KD-tree を使って、頂点同士の距離が min_dist 以上になるように押し離す。

    Args:
        obj: 対象オブジェクト (MESH)
        min_dist: 頂点同士の最低距離
        iterations: 繰り返し回数（多いほど綺麗にばらけるが時間はかかる）
    """
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    min_dist_sq = min_dist * min_dist
    vert_count = len(bm.verts)

    if vert_count == 0:
        bm.free()
        return

    for _ in range(iterations):
        # 現在の頂点位置で KD-tree 構築
        kd = kdtree.KDTree(vert_count)
        for i, v in enumerate(bm.verts):
            kd.insert(v.co, i)
        kd.balance()

        # 各頂点の移動量を蓄積するバッファ
        moves = [Vector((0.0, 0.0, 0.0)) for _ in range(vert_count)]

        # 各頂点について、近傍（min_dist 以内）を検索して押し広げる
        for i, v1 in enumerate(bm.verts):
            co1 = v1.co
            neighbors = kd.find_range(co1, min_dist)

            for co2, j, dist in neighbors:
                # 同じ頂点や、すでに処理済みのペアはスキップ
                if j <= i:
                    continue
                if dist < 1e-12:
                    continue

                # 近すぎる場合のみ押し広げる
                dist_sq = dist * dist
                if dist_sq < min_dist_sq:
                    v2 = bm.verts[j]

                    # 押し出し距離：足りない分を2頂点で半々に分ける
                    push = (min_dist - dist) * 0.5
                    direction = (co1 - co2).normalized()

                    moves[i] += direction * push
                    moves[j] -= direction * push

        # 蓄積した移動量を一括適用
        for v, m in zip(bm.verts, moves):
            v.co += m

    # メッシュに反映
    bm.to_mesh(me)
    bm.free()
    me.update()


# 実行例：アクティブオブジェクトに対して処理
obj = bpy.context.active_object
if obj and obj.type == 'MESH':
    relax_vertices_min_distance_kdtree(obj, min_dist=0.05, iterations=20)
    print("KD-tree relax done.")
else:
    print("アクティブオブジェクトが MESH ではありません。")
