# coding: UTF-8
""" Elementary cellular automata with Wolfram code.

This program was coded and tested on Mac OSX 10.13.

Jinook Oh, Cognitive Biology department, University of Vienna
September 2019.

Dependency:
    wxPython (4.0)
    Numpy (1.17)

------------------------------------------------------------------------
Copyright (C) 2019 Jinook Oh, W. Tecumseh Fitch 
- Contact: jinook.oh@univie.ac.at, tecumseh.fitch@univie.ac.at

This program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your 
option) any later version.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along 
with this program.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
"""

import sys, queue
from random import randint
from threading import Thread

import wx
import wx.lib.scrolledpanel as SPanel 
import numpy as np

DEBUG = False
__version__ = 0.1

#=======================================================================

class CellularAutomata1DFrame(wx.Frame):
    """ Printing one dimensional elementary cellular automata
    (256 rules of Wolfram)
    with a given parameters in user interface.
    """
    def __init__(self):
        if DEBUG: print("CellularAutomata1DFrame.__init__")

        ##### beginning of setting up attributes ----- 
        w_pos = (0, 25) 
        self.w_sz = [800, 600]
        self.fonts = self.setupFontsForWXApp(5)
        pi = {} 
        # top panel for UI 
        pi["tUI"] = dict(pos=(0, 0), 
                         sz=(self.w_sz[0], 40), 
                         bgCol="#cccccc", 
                         style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        tUISz = pi["tUI"]["sz"]
        # panel for drawing rule
        pi["rul"] = dict(pos=(0, tUISz[1]),
                         sz=(tUISz[0], 20),
                         bgCol="#777777",
                         style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        rulSz = pi["rul"]["sz"]
        # panel for drawing CA result
        pi["caR"] = dict(pos=(0, tUISz[1]+rulSz[1]), 
                         sz=(tUISz[0], self.w_sz[1]-tUISz[1]-rulSz[1]), 
                         bgCol="#999999", 
                         style=wx.TAB_TRAVERSAL|wx.SUNKEN_BORDER)
        self.pi = pi
        self.gbs = {} # for GridBagSizer
        self.timers = {}
        self.rn = 124
        self.rule = None
        self.caRArr = None # numpy array for CA result image 
        self.caRArrSz = (self.w_sz[1]-pi["tUI"]["sz"][1],
                         self.w_sz[0])
        self.th = None # thread 
        self.q2m = queue.Queue()  # queue to main thread

        ##### end of setting up attributes -----
        
        ### init frame
        wx.Frame.__init__(self, None, -1, "Elementary cellular automata", 
                          pos = w_pos, size = self.w_sz) 
        self.SetBackgroundColour('#333333')
        self.updateFrameSize()

        ### create (scroll) panels
        self.panel = {}
        for pk in pi.keys():
            self.panel[pk] = SPanel.ScrolledPanel(
                                                  self, 
                                                  name="%s_panel"%(pk), 
                                                  pos=pi[pk]["pos"], 
                                                  size=pi[pk]["sz"], 
                                                  style=pi[pk]["style"],
                                                 )
            self.panel[pk].SetBackgroundColour(pi[pk]["bgCol"]) 
            if pk == 'rul': 
                self.panel[pk].Bind(wx.EVT_PAINT, self.onRPaint)
            elif pk == 'caR': 
                self.panel[pk].Bind(wx.EVT_PAINT, self.onPaint)

        ##### beginning of setting up top UI panel interface -----
        bw = 5 # border width for GridBagSizer
        self.gbs["tUI"] = wx.GridBagSizer(0,0)
        row = 0
        col = 0
        sTxt = self.setupStaticText(
                                    self.panel["tUI"], 
                                    "Rule number (Wolfram's 256 rules): ", 
                                    font=self.fonts[2],
                                   )
        self.gbs["tUI"].Add(
                            sTxt, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        col += 1 
        chkB= wx.CheckBox(
                            self.panel["tUI"], 
                            -1, 
                            "Random",
                            name="randRN_chkB",
                            style=wx.CHK_2STATE
                          )
        chkB.Bind(wx.EVT_CHECKBOX, self.onCheckboxEvent)
        chkB.SetValue(False)
        self.gbs["tUI"].Add(
                            chkB, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        col += 1
        spin = wx.SpinCtrl(
                            self.panel["tUI"], 
                            -1, 
                            size=(50,-1), 
                            min=0, 
                            max=255, 
                            initial=self.rn, 
                            name='ruleN_spin',
                            style=wx.SP_WRAP|wx.SP_ARROW_KEYS,
                          )
        self.gbs["tUI"].Add(
                            spin, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        col += 1
        sTxt = self.setupStaticText(
                                    self.panel["tUI"], 
                                    'Initial line: ', 
                                    font=self.fonts[2],
                                   )
        self.gbs["tUI"].Add(
                            sTxt, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        col += 1
        cho = wx.Choice(
                            self.panel["tUI"], 
                            -1, 
                            choices=['Center seed', 'Random'],
                            name="initL_cho",
                       )
        self.gbs["tUI"].Add(
                            cho, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        col += 1
        btn = wx.Button(
                            self.panel["tUI"], 
                            -1, 
                            label="Run", 
                            name='run_btn',
                       )
        btn.Bind(wx.EVT_LEFT_DOWN, self.onMouseDown)
        self.gbs["tUI"].Add(
                            btn, 
                            pos=(row,col), 
                            flag=wx.ALIGN_CENTER_VERTICAL|wx.ALL, 
                            border=bw,
                           )
        self.panel["tUI"].SetSizer(self.gbs["tUI"])
        self.gbs["tUI"].Layout()
        self.panel["tUI"].SetupScrolling()
        ##### end of setting up top UI panel interface -----

        ### set up hot keys
        idQuit = wx.Window.NewControlId()
        self.Bind(wx.EVT_MENU, self.onClose, id=idQuit)
        accel_tbl = wx.AcceleratorTable([ 
                                    (wx.ACCEL_CMD,  ord('Q'), idQuit), 
                                        ]) 
        self.SetAcceleratorTable(accel_tbl)

        ### set up status-bar
        self.statusbar = self.CreateStatusBar(1)
        self.sbBgCol = self.statusbar.GetBackgroundColour()
        self.timers["sbTimer"] = None 

    #-------------------------------------------------------------------

    def setupFontsForWXApp(self, numFonts=5):
        """ Set up fonts for wxPython application

        Args:
            numFonts (int): Number of fonts to return.

        Returns:
            fonts (list): List of wxPython fonts.
        """
        if DEBUG: print("CellularAutomata1DFrame.setupFontsForWXApp")
        
        ### fonts setup
        if 'darwin' in sys.platform: _font = "Monaco"
        else: _font = "Courier"
        fontSz = 8
        fonts = []  # larger fonts as index gets larger 
        for i in range(numFonts):
            fonts.append(
                            wx.Font(
                                    fontSz, 
                                    wx.FONTFAMILY_SWISS, 
                                    wx.FONTSTYLE_NORMAL, 
                                    wx.FONTWEIGHT_BOLD,
                                    False, 
                                    faceName=_font,
                                   )
                        )
            fontSz += 2
        return fonts

    #-------------------------------------------------------------------
   
    def setupStaticText(self, panel, label, name=None, size=None, 
                        wrapWidth=None, font=None, fgColor=None, bgColor=None):
        """ Initialize wx.StatcText widget with more options
        
        Args:
            panel (wx.Panel): Panel to display wx.StaticText.
            label (str): String to show in wx.StaticText.
            name (str, optional): Name of the widget.
            size (tuple, optional): Size of the widget.
            wrapWidth (int, optional): Width for text wrapping.
            font (wx.Font, optional): Font for wx.StaticText.
            fgColor (wx.Colour, optional): Foreground color 
            bgColor (wx.Colour, optional): Background color 

        Returns:
            wx.StaticText: Created wx.StaticText object.
        """ 
        if DEBUG: print("CellularAutomata1DFrame.setupStaticText()")

        sTxt = wx.StaticText(panel, -1, label)
        if name != None: sTxt.SetName(name)
        if size != None: sTxt.SetSize(size)
        if wrapWidth != None: sTxt.Wrap(wrapWidth)
        if font != None: sTxt.SetFont(font)
        if fgColor != None: sTxt.SetForegroundColour(fgColor) 
        if bgColor != None: sTxt.SetBackgroundColour(bgColor)
        return sTxt

    #-------------------------------------------------------------------
    
    def updateFrameSize(self):
        """ Set window size exactly to self.w_sz without menubar/border/etc.

        Args: None

        Returns: None
        """
        if DEBUG: print("CellularAutomata1DFrame.updateFrameSize()")

        ### set window size exactly to self.w_sz 
        ### without menubar/border/etc.
        _diff = (self.GetSize()[0]-self.GetClientSize()[0], 
                 self.GetSize()[1]-self.GetClientSize()[1])
        _sz = (self.w_sz[0]+_diff[0], self.w_sz[1]+_diff[1])
        self.SetSize(_sz) 
        self.Refresh()
    
    #-------------------------------------------------------------------
    
    def receiveDataFromQueue(self, q):
        """ Receive data from Queue.

        Args:
            q (Queue): Queue to receive data.

        Returns
            rData (): Received data from the queue. 
        """
        if DEBUG: print("CellularAutomata1DFrame.receiveDataFromQueue()")

        rData = None
        try:
            if q.empty() == False: rData = q.get(False)
        except Exception as e:
            pass
        return rData    
    
    #-------------------------------------------------------------------
    
    def onCheckboxEvent(self, event):
        """ wx.CHECKBOX was clicked. 
        
        Args: event (wx.Event) 

        Returns: None
        """
        if DEBUG: print("CellularAutomata1DFrame.onCheckboxEvent()")
        
        obj = event.GetEventObject()
        objName = obj.GetName()
        if objName == "randRN_chkB": # random rule number checkbox was clicked
            # enable/disable manual rule number selection widget
            spin = wx.FindWindowByName("ruleN_spin", self.panel["tUI"])
            if obj.GetValue() == True: spin.Disable()
            else: spin.Enable()

    #-------------------------------------------------------------------

    def onMouseDown(self, event):
        """ Mouse button pressed down.
        
        Args: event (wx.Event) 

        Returns: None
        """
        if DEBUG: print("CellularAutomata1DFrame.onMouseDown()")

        obj = event.GetEventObject()
        objName = obj.GetName()
        if objName == "run_btn": self.runCAThread()
    
    #-------------------------------------------------------------------
  
    def runCAThread(self):
        """ Start a thread for running CA 

        Args: None 

        Returns: None
        """
        if DEBUG: print("CellularAutomata1DFrame.runCAThread()")

        w = self.caRArrSz[1]
        h = self.caRArrSz[0]
        caRArr = np.zeros(self.caRArrSz, np.uint8)

        ### the first line
        line = []
        cho = wx.FindWindowByName("initL_cho", self.panel["tUI"])
        initL = cho.GetString(cho.GetSelection()).lower()
        if initL == 'center seed':
            line = [0] * w
            line[int(w/2)] = 1
        elif initL == 'random':
            # randomly generate the first line
            for i in range(w): line.append(randint(0, 1))

        ### rule number
        chkB = wx.FindWindowByName("randRN_chkB", self.panel["tUI"])
        rnSpin = wx.FindWindowByName("ruleN_spin", self.panel["tUI"])
        if chkB.GetValue() == True:
            ruleNum = randint(0, 255)
            rnSpin.SetValue(ruleNum)
        else:
            ruleNum = rnSpin.GetValue()
        self.rn = ruleNum
        self.rule = '{0:08b}'.format(ruleNum)
        self.panel["rul"].Refresh() # draw rules

        ### run thread
        args = (self.q2m, w, h, caRArr, line, ruleNum,)
        self.th = Thread(target=self.runCA, args=args)
        self.th.start() # start the thread 

        ### set up a timer to check progress and receive final data 
        self.timers["updateTimer"] = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, 
                  self.updateProgress, 
                  self.timers["updateTimer"])
        self.timers["updateTimer"].Start(5)

        btn = wx.FindWindowByName("run_btn", self.panel["tUI"])
        btn.Disable()

    #-------------------------------------------------------------------

    def updateProgress(self, event):
        """ Function periodically called by a timer to check progress.
        
        Args: event (wx.Event)
        
        Returns: None
        """
        if DEBUG: print("CellularAutomata1DFrame.updateProgress()")
        
        rData = self.receiveDataFromQueue(self.q2m)

        if rData != None:
            self.showStatusBarMsg(rData[1], 5)
            p = rData[1].replace(" ","").split("/")
            if rData[1].startswith("Progress: 100"): # reached the end
                self.timers["updateTimer"].Stop()
                self.caRArr = rData[2]
                self.panel["caR"].Refresh() # draw result
                self.th.join()
                self.th = None
                btn = wx.FindWindowByName("run_btn", self.panel["tUI"])
                btn.Enable()

    #-------------------------------------------------------------------

    def runCA(self, q2m, w, h, caRArr, line, ruleNum):
        """ Make a result array of Cellular automata

        Args:
            q2m (Queue): Queue to send data back.
            w (int): Width of each line.
            h (int): Height of result image. (Number of generations.)
            caRArr (numpy.ndarray): Result image array.
            line (list): The first line.
            ruleNum (int): Rule number (among Wolfram's rules, 0-255).

        Returns: None
        """ 
        if DEBUG: print("CellularAutomata1DFrame.runCA()")
 
        rule = '{0:08b}'.format(ruleNum)
        rL = [int(x) for x in rule]

        nCC = 3 # number of cells to consider. The cell itself and its
            # two neighbors are considered in Wolfram's elementray cellular 
            # automata

        for row in range(h):
            msg = "Progrss: %.1f %%"%(float(row)/h*100)
            q2m.put(('msg', msg), True, None)
            lArr = np.asarray(line)
            caRArr[row,:] = lArr # store the current line
            nextL = [0] * w 
            for x in range(w):
                rIdx = 0 # rule index
                for ci in range(nCC):
                # cell indices
                # 0: left one, 1: self, 2: right one
                    n = int(nCC/2)
                    idx = (x + ci - n + w) % w # index to look at
                    fCI = nCC - 1 - ci # flip
                    rIdx = rIdx + (2**fCI) * line[idx]
                rIdx = (2**nCC) - 1 - rIdx # flip
                    # Rule index in Wolfram's 256 rules
                    # starts from right side, [7,6,5,4,3,2,1,0]
                nextL[x] = rL[rIdx]
            line = nextL
        q2m.put(('msg', "Progress: 100 %%", caRArr), True, None)

    #-------------------------------------------------------------------
    
    def onRPaint(self, event):
        """ Painting CA rule.

        Args: event (wx.Event)

        Returns: None
        """ 
        if DEBUG: print("CellularAutomata1DFrame.onRPaint()")
        
        evtObj = event.GetEventObject()
        dc = wx.PaintDC(evtObj)
        pBCol = self.pi["rul"]["bgCol"]
        dc.SetBackground(wx.Brush(pBCol))
        dc.Clear()

        if self.rule == None: return
        rL = [int(x) for x in self.rule]

        ### draw rule number 
        font = self.fonts[2]
        fw = font.GetPixelSize()[0]
        dc.SetFont(font)
        texts = ['Rule %i:'%(self.rn)]; coords = [(5,0)]
        fg = [wx.Colour('#000000')]; bg = [wx.Colour(pBCol)]
        dc.DrawTextList( texts, coords, fg, bg)
        lblW = coords[0][0] + fw * (len(texts[0])+1) # width of label

        ### set up for drawing rule boxes
        dc.SetPen(wx.Pen('#000000', 1))
        ph = self.pi["rul"]["sz"][1] # panel height
        m = 2 # margin between outer rectangle
        lrw = ph # outer rectangle width
        lrh = int(ph*0.66) # height
        lc = int(ph/3) # small cube size
        states = [[1,1,1], [1,1,0],
                  [1,0,1], [1,0,0],
                  [0,1,1], [0,1,0],
                  [0,0,1], [0,0,0]]
        y = 1
        ### draw rule boxes
        for i in range(len(rL)):
            x = lblW + lrw*i + m*i # starting x-coordinate for this outer rect.
            dc.SetBrush(wx.Brush(pBCol))
            dc.DrawRectangle(0+x, y, lrw, lrh) # outer rectangle
            x += 1
            ### drawing current cell states
            for si in range(len(states[i])):
                if states[i][si] == 0: col = '#ffffff'
                else: col = '#000000'
                dc.SetBrush(wx.Brush(col))
                dc.DrawRectangle(x+lc*si, y, lc, lc)
            ### drawing next cell state
            if rL[i] == 0: col = '#ffffff'
            else: col = '#000000'
            dc.SetBrush(wx.Brush(col))
            dc.DrawRectangle(x+lc, y+lc, lc, lc)
    
    #-------------------------------------------------------------------
   
    def onPaint(self, event):
        """ Painting CA result.

        Args: event (wx.Event)

        Returns: None
        """ 
        if DEBUG: print("CellularAutomata1DFrame.onPaint()")

        evtObj = event.GetEventObject()
        dc = wx.PaintDC(evtObj)
        dc.SetBackground(wx.Brush('#cccccc'))
        dc.Clear()
        if isinstance(self.caRArr, np.ndarray) == False: return 
      
        ### draw CA result
        arr = self.caRArr
        arr[arr==0] = 255
        arr[arr==1] = 0
        imgArr = np.stack( (arr, arr, arr), axis=2 )
        img = wx.ImageFromBuffer(imgArr.shape[1], imgArr.shape[0], imgArr)
        bmp = wx.Bitmap(img) # wx.BitmapFromImage(img)
        dc.DrawBitmap(bmp, 0, 0)
    
    #-------------------------------------------------------------------
    
    def showStatusBarMsg(self, txt, delTime=0):
        """ Show message on status bar
        """
        if DEBUG: print("CellularAutomata1DFrame.showStatusBarMsg")
        if self.timers["sbTimer"] != None:
            self.timers["sbTimer"].Stop()
            self.timers["sbTimer"] = None
        self.statusbar.SetStatusText(txt)
        if txt == '': bgCol = self.sbBgCol 
        else: bgCol = '#99ee99'
        self.statusbar.SetBackgroundColour(bgCol)
        if txt != '' and delTime != 0:
            self.timers["sbTimer"] = wx.CallLater(delTime,
                                                  self.showStatusBarMsg, '')

    #-------------------------------------------------------------------

    def onClose(self, event):
        """ Close this frame.

        Args: event (wx.Event)

        Returns: None
        """
        if DEBUG: print("CellularAutomata1DFrame.onClose()")

        for k in self.timers.keys():
            if self.timers[k] != None: self.timers[k].Stop()
        self.Destroy()

    #-------------------------------------------------------------------

#=======================================================================

class CA1DApp(wx.App):
    """ Initializing CellularAutomata1D app with CellularAutomata1DFrame.

    Attributes:
        frame (wx.Frame): CellularAutomata1DFrame frame.
    """
    def OnInit(self):
        if DEBUG: print("CA1DApp.OnInit()")
        self.frame = CellularAutomata1DFrame()
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

#=======================================================================

if __name__ == "__main__":
    app = CA1DApp(redirect = False)
    app.MainLoop()

