package edu.akatarin.gui;

import javax.swing.*;
import java.awt.*;

/**
 * The panel where drawing is performed.
 *
 * @author ChriZ98
 */
public class DrawPanel extends JPanel {

    private final Frame frame;

    /**
     * Initializes drawing space.
     *
     * @param frame parent frame
     */
    public DrawPanel(Frame frame) {
        super();
        this.frame = frame;
        setPreferredSize(new Dimension(Frame.DRAW_SIZE, Frame.DRAW_SIZE));

        addMouseListener(frame.getMouse());
        addMouseMotionListener(frame.getMouse());
    }

    /**
     * Paints the panel on screen.
     *
     * @param grphcs graphics to paint at
     */
    @Override
    protected void paintComponent(Graphics grphcs) {
        super.paintComponent(grphcs);
        grphcs.drawImage(frame.getImage(), 0, 0, Frame.DRAW_SIZE, Frame.DRAW_SIZE, this);
    }
}
