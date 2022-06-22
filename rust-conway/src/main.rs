use std::{vec::Vec, thread::sleep, time::Duration};
use rand::{thread_rng, Rng};
use crossbeam::thread::scope;

struct Conway {
    grid: [[bool; 100]; 100]
}

impl Conway {
    fn new() -> Self {
        Self { grid: [[false; 100]; 100] }
    }

    fn random(&mut self) -> Vec<(usize, usize)> {
        let mut changes = Vec::new();
        let mut rng = thread_rng();

        for x in 0..self.grid.len() {
            for y in 0..self.grid[x].len() {
                if rng.gen() {
                    changes.push((x, y));
                }
            }
        }

        self.apply(&changes);
        changes
    }

    fn update_chunk(&self, x1: usize, x2: usize, y1: usize, y2: usize) -> Vec<(usize, usize)> {
        let mut changes = Vec::new();

        for x in x1..x2 {
            for y in y1..y2 {
                let mut count: u8 = 0;

                for i in -1..2_i32 {
                    for j in -1..2_i32 {
                        let shifted_x = (x as i32 + i) % (self.grid.len() as i32);
                        let shifted_y = (y as i32 + j) % (self.grid[x].len() as i32);
                        let final_x = if shifted_x < 0 { self.grid.len() as i32 + shifted_x } else { shifted_x };
                        let final_y = if shifted_y < 0 { self.grid[x].len() as i32 + shifted_y } else { shifted_y };

                        if self.grid[final_x as usize][final_y as usize] && !(i == 0 && j == 0) {
                            count += 1;
                        }
                    }
                }

                if count == 3 && !self.grid[x][y] {
                    changes.push((x, y));
                } else if (count != 2 && count != 3) && self.grid[x][y] {
                    changes.push((x, y))
                }
            }
        }

        changes
    }

    /*fn update(&mut self) -> Vec<(usize, usize)> {
        let mut changes = Vec::new();

        for x in 0..self.grid.len() {
            for y in 0..self.grid[x].len() {
                let mut count: u8 = 0;

                for i in -1..2_i32 {
                    for j in -1..2_i32 {
                        let shifted_x = (x as i32 + i) % (self.grid.len() as i32);
                        let shifted_y = (y as i32 + j) % (self.grid[x].len() as i32);
                        let final_x = if shifted_x < 0 { self.grid.len() as i32 + shifted_x } else { shifted_x };
                        let final_y = if shifted_y < 0 { self.grid[x].len() as i32 + shifted_y } else { shifted_y };

                        if self.grid[final_x as usize][final_y as usize] && !(i == 0 && j == 0) {
                            count += 1;
                        }
                    }
                }

                if count == 3 && !self.grid[x][y] {
                    changes.push((x, y));
                } else if (count != 2 && count != 3) && self.grid[x][y] {
                    changes.push((x, y))
                }
            }
        }

        self.apply(&changes);
        changes
    }*/

    fn update(&mut self) -> Vec<(usize, usize)> {
        let mut changes = Vec::new();
        let half_width = self.grid.len() / 2;
        let half_height = self.grid[0].len() / 2;
        
        scope(|s| {
            let mut threads = Vec::new();

            threads.push(s.spawn(|_| {
                self.update_chunk(0, half_width, 0, half_height)
            }));
            threads.push(s.spawn(|_| {
                self.update_chunk(half_width, half_width * 2, 0, half_height)
            }));
            threads.push(s.spawn(|_| {
                self.update_chunk(0, half_width, half_height, half_height * 2)
            }));
            threads.push(s.spawn(|_| {
                self.update_chunk(half_width, half_width * 2, half_height, half_height * 2)
            }));

            for thread in threads {
                changes.append(&mut thread.join().unwrap());
            }
        }).unwrap();

        self.apply(&changes);
        changes
    }

    fn apply(&mut self, changes: &Vec<(usize, usize)>) {
        for change in changes {
            self.grid[change.0][change.1] = !self.grid[change.0][change.1];
        }
    }
}

fn main() {
    let mut conway = Conway::new();
    let initial_changes = conway.random();

    for change in initial_changes {
        println!("{},{}", change.0, change.1);
    }

    loop {
        sleep(Duration::from_millis(100));

        let changes = conway.update();

        for change in changes {
            println!("{},{}", change.0, change.1);
        }
    }
}
