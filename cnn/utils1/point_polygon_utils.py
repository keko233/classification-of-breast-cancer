# @Time    : 2018.10.12
# @Author  : kawa Yeung
# @Licence : bio-totem


import math

from shapely import affinity
from shapely.geometry import Point, Polygon, LinearRing


class PointPolygon:
    def point(self, coordinate):
        """
        generate shapely point object
        :param coordinate: coordinate, python tuple: (x, y)
        :return:
        """

        return Point(coordinate)

    def polygon(self, coordinates):
        """
        generate shapely polygon object
        :param coordinates:  list of coordinate, [(x1, y1), (x2, y2)]
        :return:
        """

        return Polygon(coordinates)

    def ellipse(self, point1, point2):
        """
        generate ellipse polygon object

        ellipse = ((x_center, y_center), (a, b), angle):
            (x_center, y_center): center point (x,y) coordinates,
            (a,b): the two semi-axis values (along x, along y),
            angle: angle in degrees between x-axis of the Cartesian base and the corresponding semi-axis.

        :param point1: the point to inscribed ellipse
        :param point2: the point to inscribed ellipse
        :return: ellipse of polygon object
        ref: https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely
        """

        x_center, y_center = (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2
        a, b = abs(point1[0] - point2[0]) / 2, abs(point1[1] - point2[1]) / 2
        ellipse = ((x_center, y_center), (a, b), 90)
        circ = Point(ellipse[0]).buffer(1)
        ell = affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
        elrv = affinity.rotate(ell, 90 - ellipse[2])

        return elrv

    def polygon_bound(self, polygon):
        """
        get the boundary of the polygon, that is LinearRing
        :param polygon: polygon
        :return:
        """

        return LinearRing(list(polygon.exterior.coords))

    def point_pylygon_position(self, point, polygon):
        """
        check the point is inside or outside the polygon
        :param point: the point
        :param polygon: the polygon
        :return: true -- point inside polygon, false -- point outside polygon (include the boundary)
        """

        return point.within(polygon)

    def point_polygon_distance(self, point, polygon):
        """
        get the distance from point to polygon
        :param point: the point
        :param polygon: the polygon
        :return:
        """

        linearRing = self.polygon_bound(polygon)

        return point.distance(linearRing)


def point_position(coordinate, region):
    """
    check the point is inside or outside the region
    :param coordinate: the point to be checked, python tuple: (X, Y)
    :param region: the region, python list: [(X1, Y1), (X2, Y2)]
    :return: (boolean, float) -- > (false[outside] / true[inside], distance))
    """

    point_polygon = PointPolygon()
    point = point_polygon.point(coordinate)
    polygon = point_polygon.polygon(region)
    position = point_polygon.point_pylygon_position(point, polygon)
    distance = point_polygon.point_polygon_distance(point, polygon)

    return position, distance



