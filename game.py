import pygame
import numpy
import math
import matplotlib.pyplot

G=10000
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE =  (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255,255,0)
RED =   (255, 0, 0)
LIGHT_GRAY =  (240, 240, 240)
DARK_GRAY = (128, 128, 128)
DEEP_DARK_GRAY = (60, 60, 60)
width, height = 800, 800

class PolygonModel():
    def __init__(self,points):
        self.points = points
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.draw_center = False

    def transformed(self):
        return [(self.x+v[0],self.y+v[1]) for v in self.points]

    def move(self, milliseconds):
        self.vx, self.vy = self.ax * milliseconds / 1000.0 + self.vx, self.ay * milliseconds / 1000.0 + self.vy
        dx, dy = self.vx * milliseconds / 1000.0, self.vy * milliseconds / 1000.0
        self.x, self.y = (self.x+dx,self.y+dy)

    def does_collide(self, other_poly):
        return False

class circle(PolygonModel):
    def __init__(self,center,radius, resolution):
        points=[(radius*math.cos(math.pi*2/resolution*i),radius*math.sin(math.pi*2/resolution*i)) for i in range(resolution)]
        super().__init__(points)
        self.radius=radius
        self.center=center
        self.x = center[0]
        self.y = center[1]
    def does_collide(self, other_poly):
        if (other_poly.radius+self.radius)**2>(self.x-other_poly.x)**2+(self.y-other_poly.y)**2:
            return True
        return False

class Ball(circle):
    def __init__(self, center, radius, velocity, resolution=20):
        super().__init__(center,radius,resolution)
        self.vx = velocity[0]
        self.vy = velocity[1]

class Planet(circle):
    def __init__(self, center, radius, mass, resolution=20):
        super().__init__(center,radius,resolution)
        self.vx = 0
        self.vy = 0
        self.mass=mass
    
    def gravityforce(self, other, milliseconds):
        dir=(self.x-other.x,self.y-other.y)
        dist=math.sqrt(dir[0]**2+dir[1]**2)
        dir=(dir[0]/dist,dir[1]/dist)
        k=G*self.mass/(dist**2)
        force=(k*dir[0],k*dir[1])
        other.ax+=force[0]* milliseconds / 1000.0
        other.ay+=force[1]* milliseconds / 1000.0

class Goal(circle):
    def __init__(self, center, radius, resolution=20):
        super().__init__(center,radius,resolution)
        self.vx = 0
        self.vy = 0

class Asteroid(PolygonModel):
    def __init__(self,center, asteroidsize, vertexnum):
        points=[]
        temp=numpy.random.rand()*2*math.pi
        (numpy.random.rand()/2+1/2)
        for i in range(vertexnum):
            points.append((asteroidsize*math.cos(temp),asteroidsize*math.sin(temp)))
            temp+=2*math.pi/vertexnum
        super().__init__(points)
        self.x = center[0]
        self.y = center[1]
        self.radius = asteroidsize

class AsteroidBelt(PolygonModel):
    def __init__(self,center,radius, depth, centralangle, angularvelocity, asteroidnum, originangle=0, asteroidminsize=0.2, asteroidmaxsize=0.5,resolution=10):
        self.center=center
        self.radius=radius
        self.depth=depth
        self.angularvelocity=angularvelocity
        self.originalangle=originangle
        self.centralangle=centralangle
        self.asteroidlist=[]
        points=[]
        for i in range(resolution+1):
            theta=centralangle*(i/resolution)+originangle
            points.append(((radius+depth/2)*math.cos(theta),(radius+depth/2)*math.sin(theta)))
        for i in range(resolution,-1,-1):
            theta=centralangle*(i/resolution)+originangle
            points.append(((radius-depth/2)*math.cos(theta),(radius-depth/2)*math.sin(theta)))
        super().__init__(points)
        
        self.x = center[0]
        self.y = center[1]

        for i in range(asteroidnum):
            d=numpy.random.rand()*depth
            r=radius+d-depth/2
            theta=centralangle*(i/(asteroidnum-1))+originangle
            cx=r*math.cos(theta)
            cy=r*math.sin(theta)
            asteroidsize=numpy.random.rand()*(asteroidmaxsize-asteroidminsize)+asteroidminsize
            asteroid=Asteroid((cx+self.x,cy+self.y), asteroidsize, 5)
            self.asteroidlist.append(asteroid)
    
    def does_collide(self, other_poly):
        maxr=self.radius+self.depth/2+other_poly.radius
        minr=self.radius-self.depth/2-other_poly.radius
        dir=(other_poly.x-self.x,other_poly.y-self.y)
        dist=math.sqrt(dir[0]*dir[0]+dir[1]*dir[1])

        if minr<=dist<=maxr:
            for asteroid in self.asteroidlist:
                if other_poly.does_collide(asteroid):
                    return True
        return False
    
class Map():
    def __init__(self,ball,goal,planets,asteroidBelts,nextMap,level):
        self.ball=ball
        self.goal=goal
        self.planets=planets
        self.asteroidBelts=asteroidBelts
        self.nextMap=nextMap

        self.state=0
        self.power =0.5
        self.maxpower=10
        self.dirAngle = 0.5
        self.sensivity = 1
        self.trajectory=[(ball.x,ball.y)]
        self.level=level
        self.Manual=False
        self.key_title=["space : launch","R : reset","Left/Right : adjust angle","Up/Down : adjust power","Shift + arrow : Fine Mode","M : See Gravity Map","Press Tab to view Manual"]

    def InputControl(self,milliseconds):
        keys = pygame.key.get_pressed()
        if self.state==0:
            if keys[pygame.K_LSHIFT]:
                self.sensivity=0.1
            else:
                self.sensivity=1

            if keys[pygame.K_UP]:
                self.power += self.sensivity * milliseconds/1000
                if self.power>1:
                    self.power=1
            elif keys[pygame.K_DOWN]:
                self.power -= self.sensivity * milliseconds/1000
                if self.power<0:
                    self.power=0

            if keys[pygame.K_RIGHT]:
                self.dirAngle += self.sensivity * milliseconds/1000
                if self.dirAngle>1:
                    self.dirAngle-=1
                
            elif keys[pygame.K_LEFT]:
                self.dirAngle -= self.sensivity * milliseconds/1000
                if self.dirAngle<0:
                    self.dirAngle+=1

            if keys[pygame.K_SPACE]:
                self.ball.ax=0
                self.ball.ay=0
                self.ball.vx=-self.maxpower*self.power*math.cos(self.dirAngle*math.pi*2)
                self.ball.vy=self.maxpower*self.power*math.sin(self.dirAngle*math.pi*2)
                self.state=1

            if keys[pygame.K_m]:
                plot_scalar_field(lambda x,y : gravity(x,y,self),self,-10,10,-10,10) 
            
            if keys[pygame.K_TAB]:
                self.Manual=True
            else:
                self.Manual=False
        
        if self.state==1 or self.state==2:
            if keys[pygame.K_r]:
                self.state=0
                self.ball.ax=0
                self.ball.ay=0
                self.ball.x=self.ball.center[0]
                self.ball.y=self.ball.center[1]
                self.trajectory=[(self.ball.x,self.ball.y)]

    def event(self,milliseconds):
        for planet in self.planets:
            if self.ball.does_collide(planet):
                self.state=2
        for asteroidbelt in self.asteroidBelts:
            if asteroidbelt.does_collide(self.ball):
                self.state=2
        if self.goal.does_collide(self.ball):
            self.state=3

        if self.state==1:
            self.ball.ax=0
            self.ball.ay=0
            for planet in self.planets:
                planet.gravityforce(self.ball, milliseconds)
            self.ball.move(milliseconds)
            self.trajectory.append((self.ball.x,self.ball.y))

    def drawMap(self,screen):
        screen.fill(BLACK)

        draw_grid(screen)

        draw_poly(screen,self.ball,(0,50,255))
        for planet in self.planets:
            if planet.mass>0:
                draw_poly(screen,planet,(160,180,160))
            else:
                draw_poly(screen,planet,(245,245,255))

        for asteroidbelt in self.asteroidBelts:
            for asteroid in asteroidbelt.asteroidlist:
                draw_poly(screen,asteroid,(125,125,125))

        draw_poly(screen,self.goal,(240,240,0))

        if self.state==0:
            v=(-5*self.power*math.cos(self.dirAngle*math.pi*2)+self.ball.x,5*self.power*math.sin(self.dirAngle*math.pi*2)+self.ball.y)
            draw_segment(screen,(self.ball.x,self.ball.y), v)
        else:
            for i in range(len(self.trajectory)-1):
                draw_segment(screen,self.trajectory[i],self.trajectory[i+1],color=BLUE)

    def UI(self,screen,myFont):
        level_title= myFont.render("Level "+str(self.level), True, YELLOW)
        screen.blit(level_title, [10, 10])
        if self.Manual:
            for i in range(6):
                temp_title= myFont.render(self.key_title[i], True, YELLOW)
                screen.blit(temp_title, [500, 10+40*i])
        else:
            temp_title= myFont.render(self.key_title[6], True, YELLOW)
            screen.blit(temp_title, [480, 10])
        
        if self.state==2:
            temp_title= myFont.render("press R to reset", True, YELLOW)
            screen.blit(temp_title, [300,400])

class endMap():
    def __init__(self,nextMap):
        self.nextMap=nextMap
        self.state=0

    def InputControl(self,milliseconds):
        return 0

    def event(self,milliseconds):
        return 0

    def drawMap(self,screen):
        screen.fill(BLACK)
        return 0

    def UI(self,screen,myFont):
        clear_title= myFont.render("You clear this game in :" +str(trial)+" "+"trial", True, YELLOW)
        text_rect=clear_title.get_rect()
        text_rect.centerx=width//2
        text_rect.centery=height//2-50
        screen.blit(clear_title, text_rect)
        clear_title= myFont.render("Game clear", True, YELLOW)
        text_rect=clear_title.get_rect()
        text_rect.centerx=width//2
        text_rect.centery=height//2
        screen.blit(clear_title, text_rect)

class startMap():
    def __init__(self,nextMap):
        self.nextMap=nextMap
        self.state=0

    def InputControl(self,milliseconds):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            self.state=3

    def event(self,milliseconds):
        return 0

    def drawMap(self,screen):
        return 0

    def UI(self,screen,myFont):
        level_title= myFont.render("press R to new Game ", True, YELLOW)
        text_rect=level_title.get_rect()
        text_rect.centerx=width//2
        text_rect.centery=height//2
        screen.blit(level_title, text_rect)

def to_pixels(x,y):
    return (width/2 + width * x / 20, height/2 - height * y / 20)

def draw_poly(screen, polygon_model, color=BLACK):
    pixel_points = [to_pixels(x,y) for x,y in polygon_model.transformed()]
    if color==BLACK:
        pygame.draw.lines(screen, color, True, pixel_points, 2)
    else:
        pygame.draw.polygon(screen,color,pixel_points)
    if polygon_model.draw_center:
        cx, cy = to_pixels(polygon_model.x, polygon_model.y)
        pygame.draw.circle(screen, BLACK, (int(cx), int(cy)), 4, 4)

def draw_segment(screen, v1,v2,color=RED):
    pygame.draw.line(screen, color, to_pixels(*v1), to_pixels(*v2), 2)

def draw_grid(screen):
    for x in range(-9,10):
        draw_segment(screen, (x,-10), (x,10), color=DARK_GRAY)
    for y in range(-9,10):
        draw_segment(screen, (-10, y), (10, y), color=DARK_GRAY)

    draw_segment(screen, (-10, 0), (10, 0), color=DEEP_DARK_GRAY)
    draw_segment(screen, (0, -10), (0, 10), color=DEEP_DARK_GRAY)

def plot_scalar_field(f,map,xmin,xmax,ymin,ymax):
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6,6)
    ax = fig.add_subplot(projection='3d')
    fv = numpy.vectorize(f)
    
    X = numpy.linspace(xmin, xmax, 200)
    Y = numpy.linspace(ymin, ymax, 200)
    X = numpy.outer(X,numpy.ones(200))
    Y = numpy.outer(numpy.ones(200),Y)
    Z = fv(X,Y)/100000

    ax.plot_wireframe(X, Y, Z,linewidth=0.5)

    u = numpy.linspace(0, 2 * numpy.pi, 10)
    v = numpy.linspace(0, numpy.pi, 10)

    x = map.ball.radius * numpy.outer(numpy.cos(u), numpy.sin(v)) + map.ball.x
    y = map.ball.radius * numpy.outer(numpy.sin(u), numpy.sin(v)) + map.ball.y
    z = map.ball.radius * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))+2
    ax.plot_surface(x, y, z, color='b')

    x = map.goal.radius * numpy.outer(numpy.cos(u), numpy.sin(v))+map.goal.x
    y = map.goal.radius * numpy.outer(numpy.sin(u), numpy.sin(v))+map.goal.y
    z = map.goal.radius * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))+2
    ax.plot_surface(x, y, z, color='yellow')
    for planet in map.planets:
        x = planet.radius * numpy.outer(numpy.cos(u), numpy.sin(v))+planet.x
        y = planet.radius * numpy.outer(numpy.sin(u), numpy.sin(v))+planet.y
        z = planet.radius * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))+2
        if planet.mass>0:
            ax.plot_surface(x, y, z, color='dimgray')
        else:
            ax.plot_surface(x, y, z, color='aliceblue')

    u = numpy.linspace(0, 2 * numpy.pi, 5)   
    v = numpy.linspace(0, numpy.pi, 5)

    for AsteroidBelt in map.asteroidBelts:
        for asteroid in AsteroidBelt.asteroidlist:
            x = asteroid.radius * numpy.outer(numpy.cos(u), numpy.sin(v))+asteroid.x
            y = asteroid.radius * numpy.outer(numpy.sin(u), numpy.sin(v))+asteroid.y
            z = asteroid.radius * numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))+2
            ax.plot_surface(x, y, z, color='lightgray')

    
    RADIUS = 10.0  # Control this value.
    ax.set_xlim3d(-RADIUS / 2, RADIUS / 2)
    ax.set_zlim3d(-RADIUS / 2, RADIUS / 2)
    ax.set_ylim3d(-RADIUS / 2, RADIUS / 2)
    ax.set_aspect("equal")
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.show()

def gravity(x,y,map):
    sum=0
    for planet in map.planets:
        dir=(planet.x-x,planet.y-y)
        dist=math.sqrt(dir[0]**2+dir[1]**2)
        sum-=G*planet.mass/(dist+0.00001)
    return sum

endmap=endMap(0)

ball = Ball([-7,4],0.3,[0,0])
goal = Ball([7,4],0.3,[0,0])
planets=[]
planets.append(Planet([0,-8], 2, -2))
asteroidBelts=[]
asteroidBelts.append(AsteroidBelt((0,4.5),4,1,math.pi*2,0,16))
asteroidBelts.append(AsteroidBelt((0,4.5),2,1,math.pi*2,0,8))
map5=Map(ball,goal,planets,asteroidBelts,endmap,5)

ball = Ball([-7,-5],0.3,[0,0])
goal = Ball([7,8],0.4,[0,0])
planets=[]
planets.append(Planet([-4,-2], 1.5, 2))
planets.append(Planet([0,2], 1, 4))
planets.append(Planet([5,6], 1.2, 2))
asteroidBelts=[]
asteroidBelts.append(AsteroidBelt((-4,-2),8,1,math.pi/3,0,8,math.pi*5/12))
asteroidBelts.append(AsteroidBelt((-4,-2),7,1,math.pi/3,0,5,math.pi*5/12))
asteroidBelts.append(AsteroidBelt((-4,-2),9,1,math.pi/3,0,8,math.pi*5/12))
asteroidBelts.append(AsteroidBelt((7,-5),7,1,math.pi,0,12,math.pi/3))
asteroidBelts.append(AsteroidBelt((7,-5),5,1,math.pi,0,12,math.pi/3))
map4=Map(ball,goal,planets,asteroidBelts,map5,4)

ball = Ball([-5,-5],0.3,[0,0])
goal = Ball([3,3],0.2,[0,0])
planets=[]
planets.append(Planet([-1,-1], 2, 2))
asteroidBelts=[]
asteroidBelts.append(AsteroidBelt((-2,-2),5,1,math.pi*2/3,1,10))
asteroidBelts.append(AsteroidBelt((-2,-2),9,1,math.pi*2/3,1,10))
map3=Map(ball,goal,planets,asteroidBelts,map4,3)

ball = Ball([-5,5],0.3,[0,0])
goal = Ball([-2,2],0.2,[0,0])
planets=[]
planets.append(Planet([-3,3], 1, 1))
planets.append(Planet([0,0], 2, 2))
asteroidBelts=[]
map2=Map(ball,goal,planets,asteroidBelts,map3,2)

ball = Ball([3,-3],0.3,[0,0])
goal = Ball([-4,4],0.2,[0,0])
planets=[]
planets.append(Planet([0,0], 1, 1))
asteroidBelts=[]
map1=Map(ball,goal,planets,asteroidBelts,map2,1)

startmap=startMap(map1)
endmap.nextMap=map1
trial=0
def main():
    global trial
    pygame.init()
    myFont = pygame.font.SysFont( "arial", 30, True, False)
    screen = pygame.display.set_mode([width,height])

    pygame.display.set_caption("Game!")

    done = False
    clock = pygame.time.Clock()

    currentmap=startmap
    trialfail=False

    while not done:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        milliseconds = 20
        
        currentmap.InputControl(milliseconds)
        currentmap.event(milliseconds)
        currentmap.drawMap(screen)
        currentmap.UI(screen,myFont)



        level_title= myFont.render("Trial : "+str(trial), True, YELLOW)
        screen.blit(level_title, [10, 40])
        pygame.display.flip()
        
        if currentmap.state==3:
            currentmap=currentmap.nextMap
        
        if currentmap.state==2 or currentmap.state==1:
            trialfail=True
        else:
            if trialfail:
                trialfail=False
                trial+=1

    pygame.quit()

if __name__ == "__main__":   
    main()