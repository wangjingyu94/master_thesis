from pykdtree.kdtree import KDTree


class KNearestNeighbor(object):

    def knn(points, k=1):
        tree = KDTree(points)
        return tree.query(points, k)
