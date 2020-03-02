from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import Image_Processing as ip


'''
This class gives the user the ability to control and manipulate the game
'''
class Control:
    
    def __init__(self,browser):
        self.driver_ = None
        if browser == "chrome":
            self.driver_ = webdriver.Chrome("./chromedriver.exe")
            self.driver_.get("chrome://dino")
        elif browser == "firefox":
            self.driver_ = webdriver.Firefox(executable_path=r"./geckodriver")
            self.driver_.get("https://chromedino.com")
        self.driver_.execute_script(
            "document.getElementsByClassName('runner-canvas')[0].id='runner-canvas'")
        
        
        
    '''
    Game Control
    The name of the functions below exactly explains what each of them does
    '''
    def __start__(self):
        self.driver_.find_element_by_tag_name("body").send_keys(Keys.SPACE)
        
    def __restart__(self):
        return self.driver_.execute_script("return Runner.instance_.restart()")
    
    def __pause__(self):
        return self.driver_.execute_script("return Runner.instance_.stop()")
    
    def __resume__(self):
        return self.driver_.execute_script("return Runner.instance_.play()")
    
    def __screenshot__(self):
        screen = self.driver_.execute_script(
            "return document.getElementById('runner-canvas').toDataURL().substring(22)")
        return ip.screenshot(screen)
    
    def __get_score__(self):
        return \
        int(''.join(self.driver_.execute_script(\
        "return Runner.instance_.distanceMeter.digits")))
    
    def __game_over__(self):
        return self.driver_.execute_script("return Runner.instance_.crashed")

    def __end__(self):
        self.driver_.close()
        
        
        
        
    '''
    Dino Control
    The dino in the game either jump or duck to avoid obstacles
    '''
    def __jump__(self):
        self.driver_.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def __duck__(self):
        self.driver_.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    
