
#import "@preview/arkheion:0.1.1": arkheion, arkheion-appendices

#show: arkheion.with(
  title: "Conditional Flow Matching for MEG inversion problem",
  authors: (
    (name: "Daniil Sherki", email: "Daniil.Sherki@skoltech.ru", affiliation: "Skoltech, Sber"),
    (name: "Anna Kubyak", email: "kubyaka4@gmail.com", affiliation: "HSE"),
    (name: "Alexei Ossadtchi", email: "ossadtchi@gmail.com", affiliation: "HSE, AIRI"),
    (name: "Ivan Oseledets", email: "i.oseledets@skoltech.ru", affiliation: "Skoltech, AIRI"),
    (name: "Ekaterina Muravleva", email: "e.muravleva@skoltech.ru", affiliation: "Skoltech, Sber"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: lorem(55),
  keywords: ("MEG", "Inverse Problem", "Generative Intelligence", "Flow Matching"),
  date: "October 1, 2025",
)
#set cite(style: "chicago-author-date")
#show link: underline

= Introduction
#lorem(60)

= Problem Statement

Сущестует проблема МЭГ. 

Пусть $M$ -- число сенсоров (магнитометров), $N$ -- число вершин (вёртексов) кортикальной сетки (источников с фиксированной ориентацией), $J$ -- число макроисточников.


Тогда прямая модель МЭГ:

$ Y = G X^T + eta $

- $G in RR^(M times N)$ - lead-field матрица.  Эта  матрица  характеризует  какой  вклад  источник  единичной  амплитуды в  данной  точке  будет  вносить  в  данные  на  каждом  сенсоре,  то  есть  чувствительность сенсоров
- $X in RR^(N times T)$ - поле активности мозга
- $Y in RR^(M times T)$ - данные на сенсорах
- $eta in RR^(M times T)$ - шум


Задается $X$ следующим образом:

$ X(:,t) = sum^J_(j=1)a_j (t) phi.alt_j $

*Пространство* задается центроидами, где центроид $c_j$ -- верншина сетки. Профиль по вершинам

$ (phi.alt_j)_i = A_(0,j) exp(-frac(1,2) (frac(d(i, c_j), sigma_j))^2) bold(1){d(i,c_j) <= R_j} $


где $d(i,c_j)$ -- геодезическое расстоения по кортикальной сетке. Центроиды выбираются так, чтобы $d(c_j, c_k)$ были больше минимального расстояния. 

*Время* задает амплитудами. Амплитуда $a_j (t)$ возникает только на отрезке $[t_j^"start", t_j^"stop"]$.

// В матричной форме

// $ underbrace(S, J times N)  = vec(phi.alt_1^T, dots.v, phi.alt_n^t) $


= Inverse Problem


== Формулировка обратной задачи

Обратная задача формулируется следующим образом:

По известной МЭГ $Y$ и известным координатам центроидов истоников $с_j$ восстановить поле активности мозга $X$, то есть все решение задачи сводится к выучиванию условного распределения $pi(X|Y)$. Для этого предлагается использовать Conditional Flow Matching из @lipman2023flowmatchinggenerativemodeling.


== Методология решения обратной задачи
Вся методология решения обратной задачи с помощью CFM  взята из @sherki2025bayesianinverseproblemsmeet.

+ *Датасет*.
  
   Собирается датасет из троек $(Y, X_1, c_j)$ с помощью симуляции прямой задачи.
+ *Обучаем модель Flow Matching*. Пусть $v_t (X_t, Y, theta)$ is flow matchcing neural network model.
  
  + Для любой пары $(Y, X_1, c_j)$ из датасета сэмплируем $X_0 ~ cal(N)(0, I) in RR^(N times T)$
  + Линейно интерполируем $ X_t = (1-t) X_1 + t * X_0 $
  + Вычисляем функцию потерь путем регрессии векторных полей $ cal(L) = norm( (X_1 - X_0) - v_t (X_t, Y, theta))^2 $
  + Обновляем веса модели с помощью стохатического градиентного спуска с помощью $nabla_theta cal(L)$.
+ *Инференс обученной модели* происходит за счет решения ODE:
$ frac("d"x,"d"t) = v_t (X, Y,theta) $

// Add bibliography and create Bibiliography section
#bibliography("bibliography.bib")


// #include "appendix.typ"