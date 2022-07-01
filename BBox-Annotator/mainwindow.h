#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "annotationview.h"
#include "networkpredictor.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    AnnotationView* m_view;
    networkPredictor* m_predictor;

};
#endif // MAINWINDOW_H
