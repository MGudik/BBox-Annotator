#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "annotationview.h"
#include "networkpredictor.h"
#include "annotationscene.h"
#include <QtCore>
#include <QFileDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
    AnnotationView* m_view;
    AnnotationScene* m_scene;
    networkPredictor* m_predictor;
    std::vector<cv::Mat> m_images;

};
#endif // MAINWINDOW_H
